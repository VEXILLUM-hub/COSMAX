# -*- coding: utf-8 -*-
"""
COSMAX 향취 후보 탐색 대시보드 — Wide Layout

실행
    streamlit run app_business_dashboard.py

필요 파일 (app.py 와 같은 폴더)
    main_dataset.csv      분자표 — SMILES, 물성, 작용기, 기록된 향취
    score_matrix.csv      분자 × 향취 예측 점수 (out-of-fold)
    per_label_test.csv    향취별 성능 — 신뢰도 배지 근거
    model_bundle.joblib   (선택) 있으면 쓰고, 없으면 앱이 직접 학습해서 만든다

model_bundle.joblib 이 없어도 ② SMILES 탭이 동작한다.
첫 실행 때 앱이 모델을 학습해 같은 폴더에 저장하고, 다음부터는 그 파일을 재사용한다.
"""

import base64
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="COSMAX 향취 후보 탐색", page_icon="🧪", layout="wide")

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model_bundle.joblib"

# 최종 모델 성능 (11_향취별_성능 / 04_최종성능 시트 기준)
TEST_MACRO_AP = 0.327


# ══════════════════════════════════════════════
# 파일 찾기 — 터미널 위치와 무관하게 동작한다
# ══════════════════════════════════════════════
def find(name, alt=(), required=True):
    for cand in (name,) + tuple(alt):
        for folder in (BASE, Path.cwd()):
            p = folder / cand
            if p.exists():
                return p
        hits = list(BASE.rglob(cand))
        if hits:
            return hits[0]
    if required:
        st.error(f"**{name}** 을(를) 찾을 수 없습니다.")
        st.write(f"찾아본 위치: `{BASE}`")
        st.code("app.py\nmain_dataset.csv\nscore_matrix.csv\nper_label_test.csv")
        found = sorted(p.name for p in BASE.glob("*") if p.suffix in (".csv", ".joblib", ".py"))
        st.write("현재 폴더의 파일:", found or "없음")
        st.stop()
    return None


@st.cache_data(show_spinner="데이터 불러오는 중…")
def load_data():
    mol = pd.read_csv(find("main_dataset.csv"))
    score = pd.read_csv(find("score_matrix.csv"))
    perf = pd.read_csv(find("per_label_test.csv", alt=("per_label_test_tuned.csv",)))
    labels = [c for c in score.columns if c != "molecule_id"]
    df = mol.merge(score, on="molecule_id", suffixes=("", "_score"))
    return df, labels, perf.set_index("label")


# ══════════════════════════════════════════════
# 특징 계산 — 학습·예측이 같은 함수를 쓴다
# ══════════════════════════════════════════════
DESC_ORDER = ["MolWt", "LogP", "TPSA", "HBD", "HBA",
              "RotatableBondCount", "RingCount", "AromaticRingCount", "FractionCSP3"]


def clean_smiles(value):
    """복사/붙여넣기 과정에서 생기는 공백·따옴표를 제거한다."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip()
    # 엑셀/메신저에서 붙여넣은 따옴표 제거
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    # SMILES 내부에는 공백이 필요하지 않으므로 모든 공백 문자 제거
    s = "".join(s.split())
    return s


def parse_smiles(value):
    """
    SMILES를 RDKit Mol로 변환한다.
    '유효하지 않은 SMILES'와 'RDKit 로드 실패'를 구분해서 반환한다.
    """
    s = clean_smiles(value)
    if not s:
        return None, s, "SMILES가 비어 있습니다."
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception as e:
        return None, s, f"RDKit을 불러오지 못했습니다: {type(e).__name__}: {e}"

    try:
        m = Chem.MolFromSmiles(s, sanitize=True)
    except Exception as e:
        return None, s, f"SMILES 해석 중 오류가 발생했습니다: {type(e).__name__}: {e}"

    if m is None:
        return None, s, "RDKit이 이 문자열을 유효한 SMILES로 해석하지 못했습니다."
    return m, s, None


def featurize(smiles_list):
    """Morgan1024 + Descriptor9 = 1,033차원. 학습과 예측이 동일한 순서를 사용한다."""
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    rows, ok = [], []

    for raw in smiles_list:
        m, _, err = parse_smiles(raw)
        if m is None:
            ok.append(False)
            continue

        desc = [
            Descriptors.MolWt(m),
            Descriptors.MolLogP(m),
            Descriptors.TPSA(m),
            Lipinski.NumHDonors(m),
            Lipinski.NumHAcceptors(m),
            Lipinski.NumRotatableBonds(m),
            rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcFractionCSP3(m),
        ]
        rows.append(np.hstack([gen.GetFingerprintAsNumPy(m), desc]))
        ok.append(True)

    X = np.array(rows, dtype=float) if rows else np.zeros((0, 1033), dtype=float)
    return X, np.array(ok, dtype=bool)


@st.cache_resource(show_spinner=False)
def get_model(_df, labels):
    """
    model_bundle.joblib 이 있으면 우선 사용한다.
    없거나 읽지 못하면 로컬 데이터로 대체 모델을 학습한다.
    """
    import joblib

    if MODEL_PATH.exists():
        try:
            b = joblib.load(MODEL_PATH)
            b["_source"] = "model_bundle.joblib"
            return b
        except Exception as e:
            st.warning(
                "저장 모델을 읽지 못했습니다. 대체 모델을 새로 학습합니다. "
                f"({type(e).__name__})"
            )

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier

    prog = st.progress(0.0, text="모델을 처음 학습합니다 (최초 1회)")
    X, ok = featurize(_df["canonical_isomeric"].tolist())
    if len(X) == 0:
        prog.empty()
        raise RuntimeError("학습 가능한 유효 SMILES가 없습니다. RDKit 설치와 데이터 컬럼을 확인하세요.")

    dfx = _df.loc[ok].reset_index(drop=True)
    Y = np.array(
        [[int(l in str(s).split("|")) for l in labels]
         for s in dfx["labels_final_pipe"].fillna("")],
        dtype=int,
    )

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    models = []
    for j, lab in enumerate(labels):
        clf = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=0,
        )
        clf.fit(Xs, Y[:, j])
        models.append(clf)
        prog.progress((j + 1) / len(labels), text=f"학습 중… {lab} ({j+1}/{len(labels)})")
    prog.empty()

    bundle = {
        "scaler": scaler,
        "models": models,
        "labels": labels,
        "feature": "Morgan1024_Desc9",
        "_source": "앱에서 학습한 대체 모델",
    }
    try:
        joblib.dump(bundle, MODEL_PATH, compress=3)
    except Exception:
        pass
    return bundle


def predict_one(bundle, smiles):
    """번들 형식이 리스트든 딕셔너리(앙상블)든 모두 처리한다."""
    X, ok = featurize([smiles])
    if len(ok) == 0 or not ok[0]:
        return None

    x = bundle["scaler"].transform(X)
    models = bundle["models"]

    if isinstance(models, dict):
        per_kind = []
        for _, lst in models.items():
            per_kind.append(np.array([m.predict_proba(x)[0, 1] for m in lst]))
        return np.mean(per_kind, axis=0)

    return np.array([m.predict_proba(x)[0, 1] for m in models])


@st.cache_data(show_spinner=False)
def mol_png(smiles, width=300, height=220):
    """
    RDKit 구조 이미지를 PNG bytes로 반환한다.
    data:image/base64 HTML 대신 st.image가 직접 렌더링하도록 해
    브라우저/Streamlit 환경에서 이미지가 사라지는 문제를 줄인다.
    """
    m, _, err = parse_smiles(smiles)
    if m is None:
        return None, err
    try:
        from rdkit.Chem import Draw
        img = Draw.MolToImage(m, size=(int(width), int(height)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), None
    except Exception as e:
        return None, f"구조 이미지 생성 실패: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════
# 신뢰도 · 주의 규칙
# ══════════════════════════════════════════════
def reliability(label, perf):
    if label not in perf.index:
        return "미평가", "⚪", "모델링 대상(100회 이상)이 아닙니다", np.nan
    p10 = float(perf.loc[label, "P@10"])
    if p10 >= 0.7:
        return "높음", "🟢", f"상위 10개 중 약 {p10*10:.0f}개가 실제 정답", p10
    if p10 >= 0.4:
        return "보통", "🟡", f"상위 10개 중 약 {p10*10:.0f}개가 실제 정답", p10
    if p10 >= 0.2:
        return "낮음", "🟠", f"상위 10개 중 약 {p10*10:.0f}개만 정답 — 참고용", p10
    return "매우 낮음", "🔴", "구조로 예측하기 어렵습니다 — 정량 판단 금지", p10


CAUTION_RULES = [
    ("has_aldehyde", "알데하이드", "공기 중 산화되어 카복실산으로 변할 수 있어 보관 안정성 확인 필요"),
    ("has_thiol", "티올", "산화되어 이황화물로 이량화될 수 있음. 미량으로도 향 지배력이 큼"),
    ("has_primary_amine", "1차 아민", "알데하이드와 공존 시 시프 염기 형성 가능"),
    ("has_phenol", "페놀", "금속이온·산화 조건에서 변색 가능"),
    ("has_lactone", "락톤", "강염기 조건에서 개환·가수분해 가능"),
    ("has_ester", "에스터", "강산·강염기 조건에서 가수분해 가능"),
    ("has_disulfide", "이황화물", "환원 조건에서 티올로 분해 가능"),
    ("has_alkene", "알켄", "산화·중합 가능성. 장기 보관 시 확인 필요"),
]


def cautions(row):
    return [(n, m) for c, n, m in CAUTION_RULES if c in row.index and row[c] > 0]


# ══════════════════════════════════════════════
# 업무용 시스템 스타일
# ══════════════════════════════════════════════
st.markdown("""
<style>
  :root {
      --bg: #08111f;
      --panel: #0d1a2c;
      --panel-2: #101f34;
      --panel-3: #142640;
      --line: #274a73;
      --line-soft: rgba(99, 155, 214, .22);
      --text: #f3f7fb;
      --muted: #94a8bf;
      --accent: #2f8cff;
      --green: #50d17a;
      --amber: #ffb547;
      --red: #ff675f;
  }

  .stApp {
      background:
        radial-gradient(circle at 72% -10%, rgba(42, 116, 206, .16), transparent 32%),
        linear-gradient(180deg, #08111f 0%, #091321 100%);
  }

  .block-container {
      padding-top: 1.05rem;
      padding-left: 1.35rem;
      padding-right: 1.35rem;
      padding-bottom: 1.3rem;
      max-width: 1840px;
  }

  header[data-testid="stHeader"] {
      background: rgba(8, 17, 31, .88);
  }

  /* 시스템 헤더 */
  .system-header {
      display: grid;
      grid-template-columns: 150px 1fr auto;
      align-items: center;
      gap: 20px;
      padding: 12px 4px 16px 4px;
      border-bottom: 1px solid var(--line-soft);
      margin-bottom: 4px;
  }
  .brand {
      font-size: 1.7rem;
      font-weight: 900;
      letter-spacing: .02em;
      color: white;
      border-right: 1px solid var(--line-soft);
      padding-right: 20px;
  }
  .system-title {
      font-size: 1.63rem;
      font-weight: 850;
      color: white;
      line-height: 1.05;
  }
  .system-subtitle {
      color: var(--muted);
      font-size: .82rem;
      margin-top: 7px;
  }
  .system-status {
      color: var(--muted);
      font-size: .78rem;
      white-space: nowrap;
  }

  /* 탭을 내비게이션처럼 */
  div[data-baseweb="tab-list"] {
      gap: 10px;
      border-bottom: 1px solid var(--line-soft);
      margin-bottom: 12px;
  }
  button[data-baseweb="tab"] {
      height: 44px;
      padding-left: 18px;
      padding-right: 18px;
      border-radius: 9px 9px 0 0;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
      background: rgba(47, 140, 255, .12);
  }

  /* Streamlit 입력/카드 */
  div[data-testid="stVerticalBlockBorderWrapper"] {
      background: linear-gradient(180deg, rgba(17, 33, 55, .96), rgba(12, 26, 45, .96));
      border: 1px solid var(--line-soft) !important;
      border-radius: 10px !important;
  }

  .section-title {
      display:flex;
      align-items:center;
      gap:8px;
      margin: 3px 0 10px 0;
      font-size: 1.06rem;
      font-weight: 800;
      color: var(--text);
  }
  .section-kicker {
      color: #58a6ff;
      font-weight: 900;
  }
  .section-note {
      font-size: .76rem;
      color: var(--muted);
      margin-top: -4px;
      margin-bottom: 8px;
  }

  .kpi-icon {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: rgba(47,140,255,.16);
      border: 1px solid rgba(47,140,255,.38);
      color: #7fbbff;
      font-size: 1.05rem;
      margin-bottom: 2px;
  }

  .pill {
      display:inline-block;
      padding: 3px 8px;
      margin-right: 5px;
      margin-top: 3px;
      border-radius: 999px;
      background: rgba(50, 107, 173, .27);
      color: #d7e9ff;
      border: 1px solid rgba(76, 137, 210, .25);
      font-size: .72rem;
      font-weight: 700;
  }
  .pill-green {
      background: rgba(52, 158, 90, .22);
      color: #a7efbd;
      border-color: rgba(72, 195, 117, .27);
  }
  .pill-orange {
      background: rgba(223, 112, 48, .18);
      color: #ffbf8c;
      border-color: rgba(255, 133, 72, .28);
  }

  .score-number {
      color: var(--green);
      font-size: 1.42rem;
      font-weight: 850;
      margin-bottom: 3px;
  }
  .tiny {
      color: var(--muted);
      font-size: .78rem;
  }
  .muted {
      color: var(--muted);
  }

  .detail-score {
      display:flex;
      align-items:center;
      gap:10px;
      margin-bottom:4px;
  }
  .detail-score strong {
      color: var(--green);
      font-size:2rem;
      font-weight:900;
  }
  .grade-high {
      background:rgba(54, 176, 92, .25);
      border:1px solid rgba(80, 215, 120, .32);
      color:#baf4c9;
      border-radius:999px;
      padding:4px 10px;
      font-size:.78rem;
      font-weight:800;
  }

  .warning-box {
      margin-top: 8px;
      padding: 12px 14px;
      border-radius: 8px;
      border: 1px solid rgba(255, 181, 71, .35);
      background: rgba(255, 181, 71, .06);
      color: #f6d39a;
      font-size: .84rem;
      line-height: 1.55;
  }

  .footer-system {
      padding: 10px 14px;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      background: rgba(16, 31, 52, .8);
      color: #a8bdd3;
      font-size: .80rem;
      margin-top: 12px;
  }

  /* 기본 컴포넌트 톤 정리 */
  div[data-testid="stMetric"] {
      background: transparent;
  }
  div[data-testid="stMetricLabel"] p {
      color: #b9c9d9;
      font-weight: 700;
  }
  div[data-testid="stMetricValue"] {
      font-size: 1.58rem;
      color: white;
  }
  div[data-testid="stHorizontalBlock"] {
      gap: .72rem;
  }
  .stButton > button {
      border-radius: 7px;
  }
  .stDownloadButton > button {
      border-radius: 7px;
  }

  @media (max-width: 1100px) {
      .system-header {
          grid-template-columns: 1fr;
      }
      .brand {
          border-right: 0;
      }
      .system-status {
          display:none;
      }
  }
</style>
""", unsafe_allow_html=True)

df, LABELS, PERF = load_data()

st.markdown(
    """
    <div class="system-header">
      <div class="brand">COSMAX</div>
      <div>
        <div class="system-title">COSMAX 향취 후보 탐색 대시보드</div>
        <div class="system-subtitle">
          분자 구조 기반으로 주요 향취를 예측하고, 목표 향취에 적합한 후보 분자를 우선순위화합니다.
        </div>
      </div>
      <div class="system-status">R&D Decision Support · 연구원 참고용</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["⌕ 후보 탐색", "⚗ SMILES → 향취 예측", "⌁ 모델 성능"])


# ══════════════════════════════════════════════
# 탭 1 — 업무용 후보 탐색 화면
# ══════════════════════════════════════════════
with tab1:
    # 검색 조건
    st.markdown('<div class="section-title"><span class="section-kicker">▾</span>검색 조건</div>', unsafe_allow_html=True)

    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([2.1, 1.0, 1.15, 1.5])

        with f1:
            default = LABELS.index("FRUITY") if "FRUITY" in LABELS else 0
            target = st.selectbox("1. 목표 향취 선택", LABELS, index=default)

        with f2:
            topn = st.number_input(
                "2. 표시 개수",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
            )

        with f3:
            only_new = st.checkbox(
                "3. 미기록 후보만",
                value=False,
                help="DB에 해당 향취가 기록되지 않았지만 모델 점수가 높은 분자만 표시합니다.",
            )
            st.caption("체크 시 신규 후보 중심")

        grade, icon, note, p10 = reliability(target, PERF)
        with f4:
            st.text_input(
                "현재 향취 신뢰도",
                value=(
                    f"{icon} {grade} · P@10 {p10:.2f}"
                    if not np.isnan(p10)
                    else f"{icon} {grade}"
                ),
                disabled=True,
            )
            st.caption("향취별 Test 성능 기반")

    # 후보 데이터
    work = df.copy()
    work["점수"] = pd.to_numeric(work[target], errors="coerce").fillna(0.0)
    work["기록됨"] = (
        work["labels_final_pipe"]
        .fillna("")
        .astype(str)
        .str.split("|")
        .apply(lambda x: target in x)
    )
    if only_new:
        work = work[~work["기록됨"]]
    work = work.nlargest(int(topn), "점수").reset_index(drop=True)

    # KPI / 성능 요약 — 향취별 사진 대신 보편 아이콘 사용
    st.markdown(
        '<div class="section-title" style="margin-top:14px;"><span class="section-kicker">▥</span>모델 신뢰도 및 성능 요약</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        with st.container(border=True):
            st.markdown('<div class="kpi-icon">⌕</div>', unsafe_allow_html=True)
            st.metric("검색 향취", target)
            st.caption("선택된 목표 향취")

    with k2:
        with st.container(border=True):
            st.markdown('<div class="kpi-icon">▦</div>', unsafe_allow_html=True)
            st.metric("상위 후보 수", f"{len(work)}개")
            st.caption("현재 검색 조건에 맞는 후보")

    with k3:
        with st.container(border=True):
            st.markdown('<div class="kpi-icon">◎</div>', unsafe_allow_html=True)
            st.metric("Precision@10", "—" if np.isnan(p10) else f"{p10:.2f}")
            st.caption(note)

    with k4:
        with st.container(border=True):
            st.markdown('<div class="kpi-icon">↗</div>', unsafe_allow_html=True)
            st.metric("Test Macro AP (41향취)", f"{TEST_MACRO_AP:.3f}")
            st.caption("최종 모델 외부 Test 성능")

    # 선택 상태가 검색 대상 변경에도 안정적으로 갱신되도록 처리
    if "sel_molecule_id" not in st.session_state:
        st.session_state.sel_molecule_id = work.iloc[0]["molecule_id"] if len(work) else None

    valid_ids = set(work["molecule_id"].astype(str))
    if len(work) and str(st.session_state.sel_molecule_id) not in valid_ids:
        st.session_state.sel_molecule_id = work.iloc[0]["molecule_id"]

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # 본문
    left, right = st.columns([1.02, 1.08])

    with left:
        st.markdown('<div class="section-title">후보 분자 Ranking</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">점수는 후보 우선순위를 위한 상대 적합도이며 실제 확률이 아닙니다.</div>',
            unsafe_allow_html=True,
        )

        rank_box = st.container(height=610)
        with rank_box:
            if len(work) == 0:
                st.info("조건에 맞는 후보가 없습니다.")
            else:
                for i, r in work.iterrows():
                    selected = str(r["molecule_id"]) == str(st.session_state.sel_molecule_id)

                    with st.container(border=True):
                        c_rank, c_img, c_info, c_score = st.columns([0.42, 1.1, 3.6, 1.35])

                        with c_rank:
                            if i == 0:
                                st.markdown("### 🥇")
                            elif i == 1:
                                st.markdown("### 🥈")
                            elif i == 2:
                                st.markdown("### 🥉")
                            else:
                                st.markdown(f"### {i+1}")

                        with c_img:
                            png, img_err = mol_png(r["canonical_isomeric"], 190, 140)
                            if png:
                                st.image(png, width=145)
                            else:
                                st.caption("구조 이미지 없음")
                                if img_err:
                                    st.caption(img_err[:70])

                        with c_info:
                            badge_txt = "DB 확인" if r["기록됨"] else "모델 제안"
                            badge_cls = "pill-green" if r["기록됨"] else "pill-orange"
                            st.markdown(
                                f"**{r['molecule_id']}** "
                                f"<span class='pill {badge_cls}'>{badge_txt}</span>",
                                unsafe_allow_html=True,
                            )

                            labs = [
                                x for x in str(r["labels_final_pipe"] or "").split("|")
                                if x
                            ]
                            if labs:
                                st.markdown(
                                    "".join(f"<span class='pill'>{lab}</span>" for lab in labs[:4]),
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.caption("기록 향취 없음")

                            if st.button(
                                "상세 보기" if not selected else "선택됨",
                                key=f"pick_business_{i}_{r['molecule_id']}",
                                type="primary" if selected else "secondary",
                            ):
                                st.session_state.sel_molecule_id = r["molecule_id"]
                                st.rerun()

                        with c_score:
                            st.markdown(
                                f"<div class='score-number'>{r['점수']:.3f}</div>"
                                f"<div class='tiny'>MW {r['MolWt']:.1f} · LogP {r['LogP']:.2f}</div>",
                                unsafe_allow_html=True,
                            )
                            st.progress(float(np.clip(r["점수"], 0, 1)))

        st.download_button(
            "↓ 전체 결과 CSV 다운로드",
            work[
                [
                    "molecule_id",
                    "canonical_isomeric",
                    "점수",
                    "기록됨",
                    "labels_final_pipe",
                    "MolWt",
                    "LogP",
                    "TPSA",
                ]
            ].to_csv(index=False),
            f"candidates_{target}.csv",
            "text/csv",
            use_container_width=True,
        )

    with right:
        st.markdown('<div class="section-title">선택 분자 상세 정보</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">후보 선택 시 구조·적합도·물성·기록 향취·주의사항을 한 화면에서 검토합니다.</div>',
            unsafe_allow_html=True,
        )

        if len(work) == 0:
            st.info("왼쪽 필터를 조정해 후보를 표시하세요.")
        else:
            matches = work[
                work["molecule_id"].astype(str) == str(st.session_state.sel_molecule_id)
            ]
            r = matches.iloc[0] if len(matches) else work.iloc[0]

            with st.container(border=True):
                d_left, d_right = st.columns([1.25, 1.0])

                with d_left:
                    st.markdown("**분자 구조**")
                    png, img_err = mol_png(r["canonical_isomeric"], 560, 360)
                    if png:
                        st.image(png, caption=str(r["molecule_id"]), use_container_width=True)
                    else:
                        st.warning("분자 구조 이미지를 만들지 못했습니다.")
                        if img_err:
                            st.caption(img_err)

                    st.caption(clean_smiles(r["canonical_isomeric"]))

                with d_right:
                    st.markdown("**예측 적합도**")
                    grade_text = "높음" if r["점수"] >= 0.7 else ("보통" if r["점수"] >= 0.4 else "낮음")
                    st.markdown(
                        f"<div class='detail-score'><strong>{r['점수']:.3f}</strong>"
                        f"<span class='grade-high'>{grade_text}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.progress(float(np.clip(r["점수"], 0, 1)))

                    st.markdown("**분자 특성**")
                    a1, a2, a3 = st.columns(3)
                    a1.metric("MW", f"{r['MolWt']:.2f}")
                    a2.metric("LogP", f"{r['LogP']:.2f}")
                    a3.metric("TPSA", f"{r['TPSA']:.2f}")

                    b1, b2, b3 = st.columns(3)
                    b1.metric("HBA", f"{int(r['HBA'])}" if pd.notna(r["HBA"]) else "—")
                    b2.metric("HBD", f"{int(r['HBD'])}" if pd.notna(r["HBD"]) else "—")
                    b3.metric(
                        "Ring Count",
                        f"{int(r['RingCount'])}" if pd.notna(r["RingCount"]) else "—",
                    )

                # 상세 하단 — 예측 향취 / 제안 여부 / 기록 향취
                scores = []
                for lab in LABELS:
                    try:
                        scores.append((lab, float(r[lab])))
                    except Exception:
                        pass
                top_pred = [lab for lab, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:3]]

                c1, c2, c3 = st.columns([1.45, .75, .85])
                with c1:
                    st.markdown("**예측 향취 (상위 3개)**")
                    if top_pred:
                        st.markdown(
                            "".join(f"<span class='pill'>{lab}</span>" for lab in top_pred),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("예측 향취 없음")

                with c2:
                    st.markdown("**모델 제안 여부**")
                    st.markdown(
                        "<span class='pill pill-green'>DB 확인</span>"
                        if r["기록됨"]
                        else "<span class='pill pill-orange'>모델 제안</span>",
                        unsafe_allow_html=True,
                    )

                with c3:
                    st.markdown("**기록된 향취**")
                    rec = str(r["labels_final_pipe"] or "없음").replace("|", " · ")
                    st.caption(rec)

                cs = cautions(r)
                if cs:
                    with st.expander(
                        f"⚠️ 구조상 주의 {len(cs)}건 · 화학 규칙 기반 / 실험 검증 아님",
                        expanded=False,
                    ):
                        for n, msg in cs:
                            st.write(f"- **{n}** — {msg}")
                else:
                    st.caption("구조상 특기할 주의사항 없음")

                st.markdown(
                    """
                    <div class="warning-box">
                    <b>참고사항</b><br>
                    본 예측은 분자 구조와 물성 정보를 이용한 모델 기반 추정치입니다.
                    최종 향취 판단은 관능평가와 안전성·규제 검토를 통해 확인해야 합니다.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        """
        <div class="footer-system">
          ⓘ 모델은 분자 구조와 물성 정보를 기반으로 향취 후보를 우선순위화합니다.
          실제 사용 전 관능평가 및 안전성 검토를 반드시 수행하십시오.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# 탭 2 — SMILES 입력
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-label">신규 분자 입력</div>', unsafe_allow_html=True)

    EXAMPLES = {
        "게라니올 (장미 계열)": "CC(C)=CCCC(C)=CCO",
        "리모넨 (시트러스 계열)": "CC1=CCC(CC1)C(C)=C",
        "벤즈알데하이드 (아몬드 계열)": "O=Cc1ccccc1",
        "바닐린 (바닐라 계열)": "COc1cc(C=O)ccc1O",
        "직접 입력": "",
    }

    e1, e2 = st.columns([1, 2.2])
    with e1:
        pick = st.selectbox("예시 선택", list(EXAMPLES.keys()), key="smiles_example")
    with e2:
        default_smiles = EXAMPLES[pick] if EXAMPLES[pick] else ""
        smiles = st.text_input(
            "SMILES",
            value=default_smiles,
            placeholder="예: CC(C)=CCCC(C)=CCO",
            key=f"smiles_input_{pick}",
        )

    go = st.button("예측 실행", type="primary", use_container_width=False)

    # 입력 즉시 파싱 상태를 보여줘 '유효하지 않음' 원인을 구분
    mol, cleaned_smiles, parse_err = parse_smiles(smiles)
    if smiles:
        if mol is not None:
            st.caption(f"✅ RDKit 구조 인식 성공 · 정리된 SMILES: `{cleaned_smiles}`")
        else:
            st.caption(f"⚠️ {parse_err}")

    if go:
        if mol is None:
            st.error(parse_err or "SMILES를 해석하지 못했습니다.")
            st.info(
                "예시 SMILES도 동일하게 실패한다면 SMILES 문법 문제가 아니라 "
                "현재 Python 환경의 RDKit 설치/로딩 문제일 가능성이 큽니다."
            )
        else:
            try:
                bundle = get_model(df, LABELS)
                t0 = time.time()
                probs = predict_one(bundle, cleaned_smiles)
            except Exception as e:
                st.error(f"모델 예측 단계에서 오류가 발생했습니다: {type(e).__name__}: {e}")
                probs = None

            if probs is not None:
                st.caption(
                    f"모델 출처: {bundle.get('_source', '파일')} · "
                    f"Feature {bundle.get('feature', 'Morgan1024_Desc9')} · "
                    f"예측 {time.time()-t0:.2f}초"
                )

                c1, c2 = st.columns([1, 2])
                with c1:
                    png, img_err = mol_png(cleaned_smiles, 380, 300)
                    if png:
                        st.image(png, caption="RDKit 2D 구조", use_container_width=True)
                    else:
                        st.warning(img_err or "구조 이미지를 생성하지 못했습니다.")

                with c2:
                    order = np.argsort(-probs)[:8]
                    rows = []
                    for idx in order:
                        lab = bundle["labels"][idx]
                        g, ic, _, _ = reliability(lab, PERF)
                        rows.append(
                            {
                                "향취": lab,
                                "적합도": round(float(probs[idx]), 3),
                                "신뢰도": f"{ic} {g}",
                            }
                        )
                    st.markdown("##### 예상 향취 Top-8")
                    st.dataframe(
                        pd.DataFrame(rows),
                        hide_index=True,
                        use_container_width=True,
                        height=310,
                    )

                st.caption(
                    "점수는 보정되지 않은 상대적 적합도이며 실제 확률이 아닙니다. "
                    "신뢰도가 낮은 향취는 참고용으로만 사용하세요."
                )

    st.divider()
    st.caption(
        "배포본에서는 가능한 한 학습 완료된 `model_bundle.joblib`을 함께 두는 것을 권장합니다. "
        "파일이 없으면 현재 코드는 대체 모델을 최초 1회 학습합니다."
    )


# ══════════════════════════════════════════════
# 탭 3 — 모델 성능
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-label">향취별 예측 성능 (Test)</div>', unsafe_allow_html=True)

    show = PERF.reset_index()[["label", "support", "AP", "F1", "P@10"]].round(3)
    show["신뢰도"] = [f"{reliability(l, PERF)[1]} {reliability(l, PERF)[0]}" for l in show["label"]]
    show = show.sort_values("P@10", ascending=False)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("모델링 향취", f"{len(show)}개")
    m2.metric("P@10 0.5 이상", f"{int((show['P@10'] >= 0.5).sum())}개")
    m3.metric("Test Macro AP", f"{TEST_MACRO_AP:.3f}")
    m4.metric("평균 P@10", f"{show['P@10'].mean():.3f}")

    st.dataframe(show, hide_index=True, use_container_width=True, height=420)
    st.bar_chart(show.set_index("label")["P@10"], height=260)

    st.markdown("##### 모델 한계")
    st.markdown("""
- 원본 향취가 **최대 3개까지만** 기록되어 있어 "없음"이 진짜 없음이 아닙니다.
  모델이 제안한 미기록 후보가 실제로 맞을 수 있습니다.
- 향취별 편차가 큽니다. 신뢰도 🟢·🟡 향취만 정량적으로 활용하세요.
- 농도·제형·온도·다른 향료와의 혼합 효과는 반영하지 않습니다.
- 주의사항은 **구조 기반 화학 규칙**이며 실험으로 검증된 배합 금기가 아닙니다.
  규제 한도는 IFRA·화장품 규정 원문을 별도 확인하세요.
""")
