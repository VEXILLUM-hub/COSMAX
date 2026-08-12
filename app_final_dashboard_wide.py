# -*- coding: utf-8 -*-
"""
COSMAX 향취 후보 탐색 대시보드 — Wide Layout

실행
    streamlit run app.py

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
# 스타일
# ══════════════════════════════════════════════
st.markdown("""
<style>
  /* 화면 가로 공간을 더 적극적으로 사용 */
  .block-container {
      padding-top: 1.6rem;
      padding-left: 1.6rem;
      padding-right: 1.6rem;
      max-width: 1780px;
  }
  /* Ranking / 상세 영역 사이 여백을 줄여 카드 폭 확보 */
  div[data-testid="stHorizontalBlock"] { gap: 0.75rem; }
  .sec-label { font-size: 0.85rem; color: #5A6B7D; font-weight: 600;
               margin: 0.2rem 0 0.4rem 0; letter-spacing: 0.02em; }
  .rank-row { display:flex; align-items:center; gap:14px; padding:9px 12px;
              border-bottom:1px solid #EDF1F5; }
  .rank-no  { font-weight:700; color:#0B2545; min-width:74px; font-size:0.95rem; }
  .rank-sc  { color:#1B6CA8; min-width:96px; font-size:0.9rem; }
  .rank-bd  { min-width:96px; font-size:0.86rem; }
  .rank-lb  { color:#6B7C8C; font-size:0.84rem; }
  .foot { color:#7A8794; font-size:0.85rem; }
  div[data-testid="stMetricValue"] { font-size: 1.55rem; }
</style>
""", unsafe_allow_html=True)

df, LABELS, PERF = load_data()

st.markdown("## COSMAX 향취 후보 탐색 대시보드")
st.caption("분자 구조로부터 향취 후보를 우선순위화하여 관능평가 전 검토 대상을 좁힙니다. "
           "관능평가를 대체하지 않습니다.")

tab1, tab2, tab3 = st.tabs(["① 향취 → 분자 찾기", "② SMILES → 향취 예측", "③ 모델 성능"])

# ══════════════════════════════════════════════
# 탭 1
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-label">필터 / 탐색 조건</div>', unsafe_allow_html=True)

    # ── 상단 필터: 스케치와 동일한 4열 구조 ──
    f1, f2, f3, f4 = st.columns([2.2, 1.1, 1.15, 1.55])
    with f1:
        default = LABELS.index("FRUITY") if "FRUITY" in LABELS else 0
        target = st.selectbox("찾고 싶은 향취", LABELS, index=default)
    with f2:
        topn = st.number_input("표시 개수", min_value=5, max_value=50, value=15, step=5)
    with f3:
        only_new = st.checkbox(
            "미기록 후보만",
            value=False,
            help="DB에 이 향취가 기록되지 않았지만 모델 점수가 높은 분자만 표시합니다.",
        )

    grade, icon, note, p10 = reliability(target, PERF)
    with f4:
        st.text_input(
            "향취 신뢰도",
            value=(f"{icon} {grade} · P@10 {p10:.2f}"
                   if not np.isnan(p10) else f"{icon} {grade}"),
            disabled=True,
        )

    # ── 후보 데이터 ──
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

    # ── KPI: 스케치의 4개 요약 카드 ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("검색 향취", target)
    k2.metric("상위 후보", f"{len(work)}개")
    k3.metric("Precision@10", "—" if np.isnan(p10) else f"{p10:.2f}")
    k4.metric("Test Macro AP", f"{TEST_MACRO_AP:.3f}")

    st.info(f"{icon} **{target} 예측 신뢰도: {grade}** — {note}")
    st.divider()

    # ── 본문: 왼쪽 Ranking / 오른쪽 선택 분자 상세 ──
    left, right = st.columns([1.55, 1.20])

    with left:
        st.markdown("#### 후보 분자 Ranking")

        if "sel_molecule_id" not in st.session_state:
            st.session_state.sel_molecule_id = work.iloc[0]["molecule_id"] if len(work) else None

        valid_ids = set(work["molecule_id"].astype(str))
        if str(st.session_state.sel_molecule_id) not in valid_ids and len(work):
            st.session_state.sel_molecule_id = work.iloc[0]["molecule_id"]

        rank_box = st.container(height=560)
        with rank_box:
            if len(work) == 0:
                st.info("조건에 맞는 후보가 없습니다.")
            else:
                for i, r in work.iterrows():
                    selected = str(r["molecule_id"]) == str(st.session_state.sel_molecule_id)
                    with st.container(border=True):
                        c_img, c_info, c_score = st.columns([1.15, 4.15, 1.35])

                        with c_img:
                            png, img_err = mol_png(r["canonical_isomeric"], 190, 140)
                            if png:
                                st.image(png, width=150)
                            else:
                                st.caption("구조 이미지 없음")
                                if img_err:
                                    st.caption(img_err[:80])

                        with c_info:
                            badge = "🔵 DB 확인" if r["기록됨"] else "🟠 모델 제안"
                            st.markdown(f"**{i+1}위 · {r['molecule_id']}**  {badge}")
                            labs = str(r["labels_final_pipe"] or "").replace("|", " · ") or "기록 없음"
                            st.caption(labs)
                            if st.button(
                                "상세 보기" if not selected else "선택됨",
                                key=f"pick_{i}_{r['molecule_id']}",
                                type="primary" if selected else "secondary",
                                use_container_width=False,
                            ):
                                st.session_state.sel_molecule_id = r["molecule_id"]
                                st.rerun()

                        with c_score:
                            st.metric("적합도", f"{r['점수']:.3f}")
                            st.caption(
                                f"MW {r['MolWt']:.1f} · "
                                f"LogP {r['LogP']:.2f}"
                            )

        st.download_button(
            "결과 CSV 다운로드",
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
        st.markdown("#### 선택 분자 상세")

        if len(work) == 0:
            st.info("왼쪽 필터를 조정해 후보를 표시하세요.")
        else:
            matches = work[
                work["molecule_id"].astype(str) == str(st.session_state.sel_molecule_id)
            ]
            r = matches.iloc[0] if len(matches) else work.iloc[0]

            png, img_err = mol_png(r["canonical_isomeric"], 520, 360)
            if png:
                st.image(png, caption=str(r["molecule_id"]), use_container_width=True)
            else:
                st.warning("분자 구조 이미지를 만들지 못했습니다.")
                if img_err:
                    st.caption(img_err)

            st.markdown(
                f"### {r['molecule_id']}  "
                f"`적합도 {r['점수']:.3f}`  "
                f"{'🔵 DB 확인' if r['기록됨'] else '🟠 모델 제안'}"
            )
            st.caption(clean_smiles(r["canonical_isomeric"]))

            m1, m2 = st.columns(2)
            m1.metric("MolWt", f"{r['MolWt']:.1f}")
            m2.metric("LogP", f"{r['LogP']:.2f}")
            m3, m4 = st.columns(2)
            m3.metric("TPSA", f"{r['TPSA']:.1f}")
            m4.metric("고리 수", f"{int(r['RingCount'])}" if pd.notna(r["RingCount"]) else "—")

            st.write("**기록 향취** — " + (str(r["labels_final_pipe"]) or "없음"))

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

    st.divider()
    st.markdown(
        '<span class="foot"><b>핵심 사용 시나리오</b> &nbsp; '
        '향취 선택 → 후보 우선순위 → 구조·물성 확인 → 미기록 후보·주의사항 검토 '
        '→ 후속 관능/분석시험</span>',
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
