# -*- coding: utf-8 -*-
"""
COSMAX 향취 예측 대시보드

실행:
    streamlit run app.py

필요 파일 (같은 폴더):
    main_dataset.csv          분자표 (SMILES, 물성, 작용기, 기록된 향취)
    score_matrix.csv          분자 x 향취 예측 점수 (out-of-fold)
    per_label_test.csv        향취별 성능 (신뢰도 배지 근거)
    model_bundle.joblib       (선택) SMILES 입력 예측용 모델
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="COSMAX 향취 예측", page_icon="🧪", layout="wide")

# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    mol = pd.read_csv("main_dataset.csv")
    score = pd.read_csv("score_matrix.csv")
    perf = pd.read_csv("per_label_test.csv")
    labels = [c for c in score.columns if c != "molecule_id"]
    df = mol.merge(score, on="molecule_id", suffixes=("", "_score"))
    return df, labels, perf.set_index("label")


@st.cache_resource
def load_model():
    try:
        import joblib
        return joblib.load("model_bundle.joblib")
    except Exception:
        return None


def mol_image(smiles, size=(260, 200)):
    """SMILES → 구조 그림 (base64 PNG)"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        img = Draw.MolToImage(m, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# ─────────────────────────────────────────────
# 신뢰도 배지
# ─────────────────────────────────────────────
def reliability(label, perf):
    """향취별 P@10 을 근거로 신뢰도 등급을 매긴다."""
    if label not in perf.index:
        return "⚪ 미평가", "이 향취는 모델링 대상(100회 이상)이 아닙니다", 0.0
    p10 = float(perf.loc[label, "P@10"])
    if p10 >= 0.7:
        return "🟢 높음", f"상위 10개 중 약 {p10*10:.0f}개가 실제 정답", p10
    if p10 >= 0.4:
        return "🟡 보통", f"상위 10개 중 약 {p10*10:.0f}개가 실제 정답", p10
    if p10 >= 0.2:
        return "🟠 낮음", f"상위 10개 중 약 {p10*10:.0f}개만 정답. 참고용", p10
    return "🔴 매우 낮음", "이 향취는 구조로 예측하기 어렵습니다. 정량 판단 금지", p10


# ─────────────────────────────────────────────
# 구조 기반 주의 규칙
# ─────────────────────────────────────────────
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

PAIR_RULES = [
    (("has_aldehyde",), ("has_primary_amine", "has_secondary_amine"),
     "알데하이드 + 아민 → 시프 염기 형성 가능"),
    (("has_thiol",), ("has_disulfide",), "티올 + 이황화물 → 산화·환원 평형 이동 가능"),
    (("has_carboxylic_acid",), ("has_primary_amine", "has_secondary_amine"),
     "카복실산 + 아민 → 염 형성으로 휘발도 변화 가능"),
]


def cautions(row):
    out = []
    for col, name, msg in CAUTION_RULES:
        if col in row and row[col] > 0:
            out.append((name, msg))
    return out


def pair_cautions(row_a, row_b):
    out = []
    for cols_a, cols_b, msg in PAIR_RULES:
        a1 = any(row_a.get(c, 0) > 0 for c in cols_a)
        b1 = any(row_b.get(c, 0) > 0 for c in cols_b)
        a2 = any(row_b.get(c, 0) > 0 for c in cols_a)
        b2 = any(row_a.get(c, 0) > 0 for c in cols_b)
        if (a1 and b1) or (a2 and b2):
            out.append(msg)
    return out


# ─────────────────────────────────────────────
# 화면
# ─────────────────────────────────────────────
df, LABELS, PERF = load_data()
bundle = load_model()

st.title("🧪 COSMAX 향취 예측 대시보드")
st.caption("분자 구조에서 예상 향취 후보를 우선순위화하여 관능평가 전 스크리닝을 지원합니다. "
           "관능평가를 대체하지 않습니다.")

tab1, tab2, tab3 = st.tabs(["① 향취 → 분자 찾기", "② SMILES → 향취 예측", "③ 모델 성능"])

# ── 탭 1: 향취 검색 ──────────────────────────
with tab1:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        target = st.selectbox("찾고 싶은 향취", LABELS,
                              index=LABELS.index("FRUITY") if "FRUITY" in LABELS else 0)
    with c2:
        topn = st.slider("표시 개수", 5, 50, 15)
    with c3:
        only_new = st.checkbox("미기록 후보만", value=False,
                               help="DB에는 이 향취가 없는데 모델이 높게 예측한 분자")

    grade, note, p10 = reliability(target, PERF)
    st.info(f"**{target}** 예측 신뢰도: {grade} — {note}")

    work = df.copy()
    work["점수"] = work[target]
    work["기록됨"] = work["labels_final_pipe"].fillna("").str.split("|").apply(lambda x: target in x)
    if only_new:
        work = work[~work["기록됨"]]
    work = work.nlargest(topn, "점수")

    for rank, (_, r) in enumerate(work.iterrows(), 1):
        badge = "🔵 DB 확인" if r["기록됨"] else "🟠 모델 제안"
        with st.container(border=True):
            left, right = st.columns([1, 3])
            with left:
                img = mol_image(r["canonical_isomeric"])
                if img:
                    st.markdown(f'<img src="data:image/png;base64,{img}" width="230">',
                                unsafe_allow_html=True)
            with right:
                st.markdown(f"### {rank}위 · {r['molecule_id']} &nbsp; `적합도 {r['점수']:.3f}` &nbsp; {badge}")
                st.caption(r["canonical_isomeric"])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("분자량", f"{r['MolWt']:.1f}")
                m2.metric("LogP", f"{r['LogP']:.2f}")
                m3.metric("TPSA", f"{r['TPSA']:.1f}")
                m4.metric("고리 수", int(r["RingCount"]))
                st.write("**기록된 향취**: " + (r["labels_final_pipe"] or "없음"))
                cs = cautions(r)
                if cs:
                    with st.expander(f"⚠️ 구조상 주의 {len(cs)}건 (화학 규칙 기반, 실험 검증 아님)"):
                        for name, msg in cs:
                            st.write(f"- **{name}** — {msg}")

    st.download_button("결과 CSV 내려받기",
                       work[["molecule_id", "canonical_isomeric", "점수",
                             "labels_final_pipe", "MolWt", "LogP"]].to_csv(index=False),
                       f"candidates_{target}.csv", "text/csv")

# ── 탭 2: SMILES 입력 ───────────────────────
with tab2:
    smiles = st.text_input("SMILES 입력", "CC(C)=CCCC(C)=CCO")
    if st.button("예측"):
        try:
            from rdkit import Chem
            m = Chem.MolFromSmiles(smiles)
        except Exception:
            m = None

        if m is None:
            st.error("유효하지 않은 SMILES 입니다.")
        elif bundle is None:
            st.warning("`model_bundle.joblib` 이 없어 신규 예측을 할 수 없습니다. "
                       "학습 노트북에서 모델을 저장한 뒤 같은 폴더에 두세요.")
        else:
            from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
            desc = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                    Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
                    Lipinski.NumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
                    rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcFractionCSP3(m)]
            x = np.hstack([gen.GetFingerprintAsNumPy(m), desc]).reshape(1, -1)
            x = bundle["scaler"].transform(x)
            probs = np.array([mo.predict_proba(x)[0, 1] for mo in bundle["models"]])

            img = mol_image(smiles, (320, 240))
            c1, c2 = st.columns([1, 2])
            if img:
                c1.markdown(f'<img src="data:image/png;base64,{img}" width="300">',
                            unsafe_allow_html=True)
            with c2:
                order = np.argsort(-probs)[:5]
                st.subheader("예상 향취 Top-5")
                for i in order:
                    g, _, _ = reliability(bundle["labels"][i], PERF)
                    st.write(f"**{bundle['labels'][i]}** &nbsp; {probs[i]:.3f} &nbsp; {g}")
                st.caption("점수는 보정되지 않은 상대적 적합도이며 실제 확률이 아닙니다.")

# ── 탭 3: 성능 ──────────────────────────────
with tab3:
    st.subheader("향취별 예측 성능 (Test)")
    show = PERF.reset_index()[["label", "support", "AP", "F1", "P@10"]].round(3)
    show["신뢰도"] = show["label"].apply(lambda l: reliability(l, PERF)[0])
    st.dataframe(show.sort_values("P@10", ascending=False), use_container_width=True, height=420)

    st.subheader("모델 한계")
    st.markdown("""
- 원본 향취가 **최대 3개까지만** 기록되어 있어 "없음"이 진짜 없음이 아닙니다.
  모델이 제안한 미기록 후보가 실제로 맞을 수 있습니다.
- 향취별 편차가 큽니다. 신뢰도 🟢 향취만 정량적으로 활용하세요.
- 농도·제형·온도·다른 향료와의 혼합 효과는 반영하지 않습니다.
- 주의 사항은 **구조 기반 화학 규칙**이며 실험으로 검증된 배합 금기가 아닙니다.
  규제 한도는 IFRA·화장품 규정 원문을 별도 확인하세요.
""")
