import json

import os

persona_id = dict(
    A01="청소년",
    A02="청년",
    A03="중년",
    A04="노년",
    G01="남성",
    G02="여성",
    C01="기본",
)

emotion_id = dict(
    S01="가족관계",
    S02="학업 및 진로",
    S03="학교폭력/따돌림",
    S04="대인관계",
    S05="연애,결혼,출산",
    S06="진로,취업,직장",
    S07="대인관계(부부, 자녀)",
    S08="재정,은퇴,노후준비",
    S09="건강",
    S10="직장, 업무 스트레스",
    S11="건강,죽음",
    S12="대인관계(노년)",
    S13="재정",
    D01="만성질환 유",
    D02="만성질환 무",
    E10="분노",
    E11="툴툴대는",
    E12="좌절한",
    E13="짜증내는",
    E14="방어적인",
    E15="악의적인",
    E16="안달하는",
    E17="구역질 나는",
    E18="노여워하는",
    E19="성가신",
    E20="슬픔",
    E21="실망함",
    E22="비통한",
    E23="후회되는",
    E24="우울한",
    E25="마비된",
    E26="염세적인",
    E27="눈물이 나는",
    E28="낙담한",
    E29="환멸을 느끼는",
    E30="불안",
    E31="두려운",
    E32="스트레스 받는",
    E33="취약한",
    E34="혼란스러운",
    E35="당혹스러운",
    E36="회의적인",
    E37="걱정스러운",
    E38="조심스러운",
    E39="초조한",
    E40="상처",
    E41="질투하는",
    E42="배신당한",
    E43="고립된",
    E44="충격 받은",
    E45="가난한, 불우한",
    E46="희생된",
    E47="억울한",
    E48="괴로워하는",
    E49="버려진",
    E50="당황",
    E51="고립된(당황한)",
    E52="남의 시선을 의식하는",
    E53="외로운",
    E54="열등감",
    E55="죄책감의",
    E56="부끄러운",
    E57="혐오스러운",
    E58="한심한",
    E59="혼란스러운(당황한)",
    E60="기쁨",
    E61="감사하는",
    E62="신뢰하는",
    E63="편안한",
    E64="만족스러운",
    E65="흥분",
    E66="느긋",
    E67="안도",
    E68="신이 난",
    E69="자신하는",
)

TRAIN_PATH = "감성대화/Training/감성대화말뭉치(최종데이터)_Training/감성대화말뭉치(최종데이터)_Training.json"
VAL_PATH = "감성대화/Validation/감성대화말뭉치(최종데이터)_Validation/감성대화말뭉치(최종데이터)_Validation.json"
TEST_PATH = "감성대화/원천데이터/감성대화말뭉치(원천데이터)_음성데이터/test_out.json"


def clean(sample):
    profile = sample["profile"]

    persona = profile["persona"]["persona-id"]
    age, gender, _ = [persona_id[c] for c in persona.split("_")]

    speaker_idx = profile["persona-id"]
    speaker_context = profile["emotion"]["emotion-id"]

    speaker_context = speaker_context.split("_")
    context, disease, emotion = [emotion_id[c] for c in speaker_context]

    talk = sample["talk"]
    conv_id = talk["id"]["talk-id"]
    dialogue = talk["content"]

    truncated = []
    for k in ["HS01", "SS01", "HS02", "SS02", "HS03", "SS03"]:
        utterance = dialogue[k]
        if len(utterance):
            truncated.append(utterance)

    return dict(
        speaker_idx=speaker_idx,
        context=context,
        disease=disease,
        emotion=emotion,
        conv_id=conv_id,
        dialogue=truncated,
    )


def load(root):
    paths = {
        "train": os.path.join(root, TRAIN_PATH),
        "val": os.path.join(root, VAL_PATH),
        "test": os.path.join(root, TEST_PATH),
    }

    dataset = {}
    for split in paths:
        with open(paths[split]) as f:
            raw_data = json.load(f)

        cleaned = [clean(sample) for sample in raw_data]

        split_data = {}
        for k in cleaned[0]:
            split_data[k] = []

        for sample in cleaned:
            for k in sample:
                split_data[k].append(sample[k])

        dataset[split] = split_data

    return dataset
