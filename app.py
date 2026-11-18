"""app.py - ແດສບອດ Streamlit: ວິເຄາະປັດໄຈສະພາບອາກາດ ແລະ ກາຊມົນລະພິດ ທີ່ກ່ຽວຂ້ອງກັບຄ່າ PM2.5

ໄຟລ໌ທີ່ຕ້ອງຢູ່ໃນໂຟນເດີດຽວກັນ:
    app.py, inference.py, features.py, weather_causes_model.pkl, requirements.txt
ຣັນ:  streamlit run app.py
"""
import os
import re

import altair as alt
import pandas as pd
import streamlit as st

import inference as inf

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "weather_causes_model.pkl")

st.set_page_config(page_title="ວິເຄາະສາເຫດ PM2.5", page_icon="🌫️", layout="wide")

# ----------------------------------------------------------------------------
# ຄ່າຄົງທີ່
# ----------------------------------------------------------------------------
RAISE, LOWER, NORMAL, NEUTRAL = "#d1495b", "#3a86a8", "#8d99a6", "#b8c0c9"
DIR_DOMAIN = ["ເພີ່ມຝຸ່ນ", "ຊ່ວຍຫຼຸດຝຸ່ນ", "ປົກກະຕິ", "ບໍ່ຊັດເຈນ"]
DIR_RANGE = [RAISE, LOWER, NORMAL, NEUTRAL]

MONTHS = ["ມັງກອນ", "ກຸມພາ", "ມີນາ", "ເມສາ", "ພຶດສະພາ", "ມິຖຸນາ",
          "ກໍລະກົດ", "ສິງຫາ", "ກັນຍາ", "ຕຸລາ", "ພະຈິກ", "ທັນວາ"]

# ຊື່ລາວ, ຫົວໜ່ວຍ, ກຸ່ມ
FACTORS = {
    "TEMP": ("ອຸນຫະພູມ", "°C", "ສະພາບອາກາດ"),
    "PRES": ("ຄວາມກົດອາກາດ", "hPa", "ສະພາບອາກາດ"),
    "DEWP": ("ຄວາມຊື້ນ (ຈຸດນ້ຳຄ້າງ)", "°C", "ສະພາບອາກາດ"),
    "WSPM": ("ຄວາມໄວລົມ", "m/s", "ສະພາບອາກາດ"),
    "NO2": ("NO₂", "µg/m³", "ກາຊມົນລະພິດ"),
    "CO": ("CO", "µg/m³", "ກາຊມົນລະພິດ"),
    "O3": ("O₃", "µg/m³", "ກາຊມົນລະພິດ"),
}

# ຄ່າສຳພັນກັບ PM2.5 ຈາກ EDA ໃນ notebook (ເຊວ 21) ໃຊ້ກຳນົດທິດ "ເພີ່ມຝຸ່ນ / ຊ່ວຍຫຼຸດຝຸ່ນ"
CORR_WITH_PM25 = {"CO": 0.76, "NO2": 0.67, "DEWP": 0.12, "PRES": 0.01,
                  "O3": -0.10, "TEMP": -0.11, "WSPM": -0.27}
MIN_CORR = 0.05      # |corr| ຕ່ຳກວ່ານີ້ ຖືວ່າທິດບໍ່ຊັດເຈນ
MIN_Z = 0.5          # |z| ຕ່ຳກວ່ານີ້ ຖືວ່າຢູ່ໃນລະດັບປົກກະຕິ

# ເກນແບ່ງລະດັບ PM2.5 ຕາມ notebook (ເກນຂອງໂປຣເຈັກ ບໍ່ແມ່ນມາດຕະຖານ AQI ທາງການ)
LEVELS = [
    (50, "ດີ", "#2a9d8f"),
    (100, "ປານກາງ", "#c9a227"),
    (150, "ເລີ່ມມີຜົນຕໍ່ກຸ່ມສ່ຽງ", "#f4842c"),
    (200, "ບໍ່ດີຕໍ່ສຸຂະພາບ", "#d1495b"),
    (float("inf"), "ອັນຕະລາຍ", "#8c1c2b"),
]

HEALTH_ADVICE = {
    "ດີ": [
        "ກຸ່ມຄົນທົ່ວໄປ: ອາກາດຢູ່ໃນເກນດີຫຼາຍ ສາມາດເຮັດກິດຈະກຳກາງແຈ້ງ ແລະ ອອກກຳລັງກາຍໄດ້ຢ່າງເຕັມທີ່.",
        "ກຸ່ມທີ່ມີບັນຫາສຸຂະພາບທາງລະບົບຫາຍໃຈ: ສາມາດໃຊ້ຊີວິດ ແລະ ດຳເນີນກິດຈະກຳໄດ້ເປັນປົກກະຕິ ໂດຍບໍ່ມີຂໍ້ຈຳກັດ.",
        "ເປີດປະຕູ ແລະ ປ່ອງຢ້ຽມເພື່ອຮັບລະບາຍອາກາດບໍລິສຸດເຂົ້າມາ ພາຍໃນອາຄານທີ່ຢູ່ອາໄສໄດ້ສະບາຍ.",
    ],
    "ປານກາງ": [
        "ກຸ່ມຄົນທົ່ວໄປ: ສາມາດເຮັດກິດຈະກຳກາງແຈ້ງໄດ້ຕາມປົກກະຕິ ແຕ່ຄວນສັງເກດຮ່າງກາຍ.",
        "ກຸ່ມທີ່ມີບັນຫາສຸຂະພາບທາງລະບົບຫາຍໃຈ: ຄວນຫຼຸດຜ່ອນການອອກກຳລັງກາຍ ຫຼື ການອອກແຮງໜັກກາງແຈ້ງເປັນເວລາດົນ.",
        "ຖ້າມีອາການໄອ, ລະຄາຍເຄືອງຄໍ ຫຼື ຫາຍໃຈຕິດຂັດ ຄວນຫຼຸດເວລາຢູ່ກາງແຈ້ງທັນທີ.",
    ],
    "ເລີ່ມມີຜົນຕໍ່ກຸ່ມສ່ຽງ": [
        "ກຸ່ມຄົນທົ່ວໄປ: ຫຼຸດຜ່ອນການເຮັດກິດຈະກຳ ແລະ ການອອກກຳລັງກາຍກາງແຈ້ງທີ່ໃຊ້ເວລາດົນ.",
        "ກຸ່ມທີ່ມີບັນຫາສຸຂະພາບທາງລະບົບຫາຍໃຈ (ຜູ້ປ່ວຍຫອບຫືດ, ພູມແພ້, ໂລກປອດ): ຄວນງົດກິດຈະກຳກາງແຈ້ງຢ່າງເດັດຂາດ ແລະ ສວມໜ້າກາກ N95/KN95 ທຸກຄັ້ງທີ່ຈຳເປັນຕ້ອງອອກນອກ.",
        "ປິດປະຕູໜ້າຕ່າງໃຫ້ມິດຊິດ, ເປີດໃຊ້ງານເຄື່ອງຟອກອາກາດ ແລະ ກຽມຢາພົ່ນຂະຫຍາຍຫຼອດລົມໄວ້ໃຫ້ພ້ອມ.",
    ],
    "ບໍ່ດີຕໍ່ສຸຂະພາບ": [
        "ກຸ່ມຄົນທົ່ວໄປ: ຫຼີກລ່ຽງການເຮັດກິດຈະກຳທຸກປະເພດກາງແຈ້ງ, ສວມໜ້າກາກກັນຝຸ່ນຂະໜາດເລັກທຸກຄັ້ງທີ່ອອກນອກອາຄານ.",
        "ກຸ່ມທີ່ມີບັນຫາສຸຂະພາບທາງລະບົບຫາຍໃຈ: ຕ້ອງພັກຜ່ອນຢູ່ພາຍໃນອາຄານປິດທີ່ຕິດເປີດເຄື່ອງຟອກອາກາດຕະຫຼອດເວລາ ຫ້າມອອກກາງແຈ້ງເດັດຂາດ.",
        "ຫາກມີອາການແໜ້ນໜ້າເອິກ, ຫາຍໃຈຖີ່ ຫຼື ຫາຍໃຈລຳບາກຮຸນແຮງ ໃຫ້ຣີບໄປພົບແພດຊ່ຽວຊານທັນທີ.",
    ],
    "ອັນຕະລາຍ": [
        "ກຸ່ມຄົນທົ່ວໄປ: ງົດການອອກນອກອາຄານໂດຍບໍ່ຈຳເປັນ, ປິດປະຕູໜ້າຕ່າງປ້ອງກັນຝຸ່ນລະອອງເຂົ້າເຮືອນ.",
        "ກຸ່ມທີ່ມີບັນຫາສຸຂະພາບທາງລະບົບຫາຍໃຈ: ຄວາມສ່ຽງສູງຫຼາຍ ຕ້ອງລະມັດລະວັງເປັນພິເສດຂີດສຸດ ຖ້າມີອາການຜິດປົກກະຕິທາງເດີນຫາຍໃຈໃຫ້ໄປໂຮງໝໍດ່ວນ.",
        "ສວມໜ້າກາກ N95/KN95 ທີ່ແນບຊິດກັບໃບໜ້າຢ່າງເຄັ່ງຄັດຫາກຈຳເປັນຕ້ອງຢູ່ໃນພື້ນທີ່ເປີດ.",
    ],
}

# ຄຳອະທິບາຍ + ວິທີປ້ອງກັນ ເມື່ອປັດໄຈນັ້ນ "ຜັກດັນໃຫ້ຝຸ່ນສູງ" (ທິດຄົງທີ່ຕໍ່ປັດໄຈ)
FACTOR_ADVICE = {
    "WSPM": {
        "why": "ຄວາມໄວລົມຕໍ່າກວ່າປົກກະຕິ ເຮັດໃຫ້ອາກາດນິ່ງ ສະພາບອາກາດປິດ ຝຸ່ນລະອອງຈຶ່ງບໍ່ສາມາດຖືກພັດກະຈາຍອອກຈາກພື້ນທີ່ໄດ້ ຈຶ່ງເກີດການສະສົມໜາແໜ້ນ.",
        "personal": ["ຕິດຕາມພະຍາກອນອາກາດ ແລະ ຫຼີກລ່ຽງການອອກກຳລັງກາຍກາງແຈ້ງໃນຊ່ວງລົມນິ່ງ.", "ສວມໜ້າກາກກັນຝຸ່ນມາດຕະຖານເມື່ອຕ້ອງຢູ່ໃນພື້ນທີ່ເປີດແຈ້ງ."],
        "community": ["ປະກາດຂໍຄວາມຮ່ວມມືງົດການເຜົາຂີ້ເຫຍື້ອ ແລະ ຫຼຸດຜ່ອນກິດຈະກຳກໍ່ສ້າງທີ່ກໍ່ໃຫ້ເກີດຝຸ່ນລະອອງໃນຊ່ວງລົມນິ່ງ.", "ລະດົມລົດບັນທຸກສີດນ້ຳລ້າງຖະໜົນ ແລະ ພົ່ນນ້ຳເພື່ອກົດຝຸ່ນໃນເຂດຊຸມຊົນ."],
    },
    "CO": {
        "why": "ຄ່າກາສກາກບອນມອນອກຊິດ (CO) ສູງຜິດປົກກະຕິ ເຊິ່ງເປັນສັນຍານຊັດເຈນຂອງການເຜົາໄໝ້ທີ່ບໍ່ສົມບູນ ບໍ່ວ່າຈະມາຈາກທໍ່ໄອເສຍຍານພາຫະນະ ຫຼື ການເຜົາໄໝ້ຊີວະມະຫາຊົນ.",
        "personal": ["ງົດການຈູດເຜົາຖ່ານ, ຟືນ ຫຼື ຂີ້ເຫຍື້ອພາຍໃນ ແລະ ບໍລິເວນໃກ້ຄຽງທີ່ຢູ່ອາໄສ.", "ບຳລຸງຮັກສາລະບົບເຜົາໄໝ້ເຄື່ອງຈັກຍານພາຫະນະ (ລົດຍົນ/ລົດຈັກ) ໃຫ້ສະອາດສົມບູນ."],
        "community": ["ເຂັ້ມງວດກວດກາ ແລະ ຄວບຄຸມການປ່ອຍໄອເສຍຈາກລົດຍົນ ແລະ ໂຮງງານອຸດສາຫະກຳ.", "ສົ່ງເສີມການໃຊ້ພະລັງງານສະອາດ ແລະ ຫຼຸດຜ່ອນການເຜົາໄໝ້ກາງແຈ້ງ."],
    },
    "NO2": {
        "why": "ກາສໄນໂຕຣເຈນໄດອອກໄຊ (NO₂) ສູງ ເຊິ່ງສ່ວນໃຫຍ່ເກີດຈາກການຈາລະຈອນໜາແໜ້ນ ແລະ ການເຜົາໄໝ້ເຊື້ອໄຟຟອດຊິວໃນເຂດຕົວເມືອງ ເຊິ່ງເປັນຕົ້ນຕໍຂອງການເກີດຝຸ່ນຮອງ.",
        "personal": ["ຫຼຸດຜ່ອນການໃຊ້ລົດຍົນສ່ວນບຸກຄົນ ໂດຍຫັນມາໃຊ້ລະບົບຂົນສົ່ງສາທາລະນະ ຫຼື ໃຊ້ວິທີຂີ່ລົດຮ່ວມກັນ (Carpooling).", "ຫຼີກລ່ຽງການເດີນທາງຜ່ານເສັ້ນທາງທີ່ມີການຈາລະຈອນຕິດຂັດໜາແໜ້ນໃນຊົ່ວໂມງເລັ່ງດ່ວນ."],
        "community": ["ຈັດຕັ້ງມາດຕະການກວດກາສະພາບລົດຄວບຄຸມມົນລະພິດ ແລະ ຈຳກັດເຂດແລ່ນລົດບາງປະເພດ.", "ສົ່ງເສີມການໃຊ້ພາຫະນະໄຟຟ້າ (EV) ແລະ ພັດທະນາລະບົບຂົນສົ່ງມວນຊົນໃຫ້ມີປະສິດທິພາບ."],
    },
    "TEMP": {
        "why": "ອຸນຫະພູມອາກາດຕໍ່າກວ່າເກນປົກກະຕິ ມັກກ່ຽວຂ້ອງກັບປະກົດການອາກາດຜິດປົກກະຕິ ຫຼື ການຈູດເຜົາເພື່ອຄວາມອົບອຸ່ນ ເຮັດໃຫ້ເກີດຊັ້ນອາກາດປີ້ນອຸນຫະພູມ (Temperature inversion) ທີ່ກັກຝຸ່ນໄວ້ໃກ້ພື້ນດິນ.",
        "personal": ["ໃຊ້ເຄື່ອງເຮັດຄວາມອົບອຸ່ນທີ່ສະອາດ (ໄຟຟ້າ) ແທນການຈູດເຜົາຖ່ານ ແລະ ຟືນໃນອາຄານ.", "ຫຼຸດການເຜົາໄໝ້ໃນເຮືອນ ແລະ ເສີມການກັນຄວາມເຢັນ (ເຄື່ອງນຸ່ງ, ຜ້າຫົ່ມ)."],
        "community": ["ສົ່ງເສີມເຊື້ອໄຟເຮັດຄວາມອົບອຸ່ນທີ່ສະອາດ."],
    },
    "DEWP": {
        "why": "ຄວາມຊື້ນສຳພັດ ແລະ ຈຸດນ້ຳຄ້າງສູງ ເຮັດໃຫ້ອະນຸພາກຝຸ່ນລະອອງດູດຊັບຄວາມຊື້ນ ເກີດເປັນໝອກຄວັນ (Smog) ທີ່ແຂວນລອຍຄ້າງຢູ່ໃນບັນຍາກາດດົນກວ່າປົກກະຕິ.",
        "personal": ["ໃຊ້ເຄື່ອງກອງອາກາດ/ເຄື່ອງລົດຄວາມຊື້ນໃນຫ້ອງ ແລະ ເຝົ້າລະວັງອາການຫາຍໃຈ.", "ຕິດຕາມຄ່າຝຸ່ນເປັນປະຈຳໃນວັນທີ່ອາກາດຊຸ່ມ."],
        "community": ["ຄວາມຊື້ນຄວບຄຸມບໍ່ໄດ້ ຈຶ່ງຄວນຫຼຸດແຫຼ່ງປ່ອຍຝຸ່ນໃນວັນທີ່ອາກາດຊຸ່ມ."],
    },
    "PRES": {
        "why": "ຄວາມກົດອາກາດສູງຜິດປົກກະຕິ ສ້າງສະພາບແວດລ້ອມອາກາດນິ່ງ ລົມສະຫງົບ ບໍ່ມີການເຄື່ອນທີ່ ເຮັດໃຫ້ຝຸ່ນລະອອງທີ່ປ່ອຍອອກມາຖືກກັກເກັບໄວ້ໃນລະດັບພື້ນຜິວ.",
        "personal": ["ຕິດຕາມປະກາດຄຸນນະພາບອາກາດ ແລະ ຫຼີກລ່ຽງກິດຈະກຳກາງແຈ້ງໃນຊ່ວງຝຸ່ນສູງ.", "ປິດປະຕູໜ້າຕ່າງໃຫ້ສະໜິດເພື່ອປ້ອງກັນຝຸ່ນຈາກພາຍນອກລອດເຂົ້າມາ."],
        "community": ["ຫຼຸດການປ່ອຍຝຸ່ນ/ການເຜົາໄໝ້ໃນຊ່ວງທີ່ອາກາດນິ່ງ."],
    },
    "O3": {
        "why": "ກາສໂອໂຊນ (O₃) ລະດັບດິນຕໍ່າຜິດປົກກະຕິ ມັກຊີ້ວ່າມີປະຕິກິລິຍາເຄມີຊັບຊ້ອນໃນບັນຍາກາດ ບວກກັບການຖືກດຶງໄປໃຊ້ໃນຂະບວນການທຳລາຍມົນລະພິດອື່ນໆໃນຊ່ວງທີ່ມົນລະພິດໜັກ.",
        "personal": ["ຫຼຸດຜ່ອນການໃຊ້ຍານພາຫະນະສ່ວນຕົວໃນວັນທີ່ມົນລະພິດສູງ."],
        "community": ["ຄວບຄຸມການປ່ອຍ NOx ຈາກການຈະລາຈອນ ແລະ ໂຮງງານ."],
    },
}


# ----------------------------------------------------------------------------
# ໂມເດວ ແລະ ການວິເຄາະ
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="ກຳລັງໂຫຼດໂມເດວ...")
def get_bundle():
    return inf.load(MODEL_PATH)


def md_bold(text):
    """ແປງ **ຕົວໜາ** (markdown) ເປັນ <b>ຕົວໜາ</b> ເພື່ອໃຫ້ສະແດງຖືກໃນກ່ອງ HTML"""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


SEVERITY_NOTE = {
    "ເລັກນ້ອຍ": "ຢູ່ນອກຄ່າປົກກະຕິເລັກນ້ອຍ ແຕ່ຍັງຄວນຮັກສາ ແລະ ປ້ອງກັນ ເພື່ອບໍ່ໃຫ້ຝຸ່ນ PM ຂຶ້ນສູງໄປກວ່ານີ້ ແລະ ພະຍາຍາມໃຫ້ຄ່າ PM ກັບສູ່ລະດັບປົກກະຕິ ຫຼື ດີກວ່າປົກກະຕິ",
    "ປານກາງ": "ຜິດປົກກະຕິຊັດເຈນ ຄວນລົງມືຫຼຸດຕາມຄຳແນະນຳໃນແຖບ \"ວິທີປ້ອງກັນ\"",
    "ຫຼາຍ": "ຜິດປົກກະຕິຫຼາຍ ເປັນບູລິມະສິດທີ່ຄວນແກ້ໄຂກ່ອນ",
}

# ເປົ້າໝາຍການແກ້ໄຂຕໍ່ປັດໄຈ (ປັດໄຈສະພາບອາກາດຄວບຄຸມບໍ່ໄດ້ ຈຶ່ງເນັ້ນຫຼຸດຜົນກະທົບ)
GOAL = {
    "CO": "ເປົ້າໝາຍ: ຫຼຸດ CO ໃຫ້ກັບສູ່ລະດັບປົກກະຕິ (≈ {normal:.0f} {unit}) ຫຼື ຕ່ຳກວ່າ ໂດຍຫຼຸດການເຜົາໄໝ້.",
    "NO2": "ເປົ້າໝາຍ: ຫຼຸດ NO₂ ໃຫ້ກັບສູ່ລະດັບປົກກະຕິ (≈ {normal:.0f} {unit}) ຫຼື ຕ່ຳກວ່າ ໂດຍຫຼຸດການຈະລາຈອນ/ການເຜົາໄໝ້.",
    "O3": "O₃ ຕ່ຳກວ່າປົກກະຕິມັກສຳພັນກັບ NOx ທີ່ສູງ. ເປົ້າໝາຍ: ຫຼຸດ NOx ຈາກການຈະລາຈອນ/ການເຜົາໄໝ້ ເພື່ອໃຫ້ O₃ ກັບສູ່ລະດັບປົກກະຕິ (≈ {normal:.0f} {unit}).",
    "WSPM": "ລົມເປັນປັດໄຈທຳມະຊາດ ຄວບຄຸມບໍ່ໄດ້. ເປົ້າໝາຍ: ຫຼຸດການປ່ອຍຝຸ່ນ/ການເຜົາໄໝ້ໃນຊ່ວງທີ່ລົມອ່ອນ ເພື່ອບໍ່ໃຫ້ຝຸ່ນສະສົມ.",
    "TEMP": "ອຸນຫະພູມເປັນປັດໄຈທຳມະຊາດ ຄວບຄຸມບໍ່ໄດ້. ເປົ້າໝາຍ: ຫຼຸດການເຜົາໄໝ້ເພື່ອໃຫ້ຄວາມອົບອຸ່ນ ແລະ ໃຊ້ເຊື້ອໄຟທີ່ສະອາດ.",
    "DEWP": "ຄວາມຊື້ນເປັນປັດໄຈທຳມະຊາດ ຄວບຄຸມບໍ່ໄດ້. ເປົ້າໝາຍ: ຫຼຸດການປ່ອຍຝຸ່ນໃນວັນທີ່ອາກາດຊຸ່ມ ແລະ ປ້ອງກັນຕົນເອງໃນຫ້ອງ.",
    "PRES": "ຄວາມກົດອາກາດເປັນປັດໄຈທຳມະຊາດ ຄວບຄຸມບໍ່ໄດ້. ເປົ້າໝາຍ: ຫຼຸດການປ່ອຍຝຸ່ນໃນຊ່ວງທີ່ອາກາດນິ່ງ.",
}


def severity(z):
    a = abs(z)
    if a < MIN_Z:
        return "ປົກກະຕິ"
    if a < 1.0:
        return "ເລັກນ້ອຍ"
    if a < 2.0:
        return "ປານກາງ"
    return "ຫຼາຍ"


def pm_level(pm25):
    for limit, name, color in LEVELS:
        if pm25 <= limit:
            return name, color
    return LEVELS[-1][1], LEVELS[-1][2]


def analyze(pm25, hour, month):
    """ຄືນ DataFrame ລາຍປັດໄຈ: ຄ່າທຳນາຍ, z-score, ທິດ, ສັດສ່ວນ (%) ຂອງປັດໄຈທີ່ເພີ່ມຝຸ່ນ"""
    b = get_bundle()
    gain = b.get("pm25_gain", b["test_r2"])   # ຜົນຊ່ວຍຂອງ PM2.5 (ຖ້າບໍ່ມີ ໃຊ້ R² ແທນ)
    pred = inf.predict_causes(pm25, hour, month)

    rows = []
    for col, d in pred.items():
        name, unit, group = FACTORS[col]
        if isinstance(b["base_mean"].index, pd.MultiIndex):
            key = (month, hour)
        else:
            key = month
        mean = float(b["base_mean"].loc[key, col])
        std = float(b["base_std"].loc[key, col])
        z = (d["pred"] - mean) / std if std > 0 else 0.0
        r2 = float(b["test_r2"][col])

        corr = CORR_WITH_PM25[col]
        if abs(z) < MIN_Z:
            direction = "ປົກກະຕິ"
        elif abs(corr) < MIN_CORR:
            direction = "ບໍ່ຊັດເຈນ"
        elif z * corr > 0:
            direction = "ເພີ່ມຝຸ່ນ"
        else:
            direction = "ຊ່ວຍຫຼຸດຝຸ່ນ"

        weight = abs(z) * max(float(gain[col]), 0.0) if direction == "ເພີ່ມຝຸ່ນ" else 0.0

        rows.append({
            "col": col, "name": name, "unit": unit, "group": group,
            "pred": d["pred"], "low": d["low"], "high": d["high"],
            "normal": mean, "z": z, "r2": r2,
            "direction": direction, "weight": weight,
            "severity": severity(z),
        })

    df = pd.DataFrame(rows)
    total = df["weight"].sum()
    df["share"] = df["weight"] / total * 100 if total > 0 else 0.0
    return df


# ----------------------------------------------------------------------------
# ກຣາຟ (Altair ສະແດງຜົນໃນ browser ຈຶ່ງສະແດງຟອນລາວได้)
# ----------------------------------------------------------------------------
def chart_share(raising):
    d = raising.copy()
    d["label"] = d.apply(lambda r: f"{r['name']}  {r['share']:.1f}% ({r['severity']})", axis=1)
    return (
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("share:Q", title="ສັດສ່ວນ (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320, labelFontSize=13)),
            color=alt.Color("group:N", title=None,
                            scale=alt.Scale(domain=["ສະພາບອາກາດ", "ກາຊມົນລະພິດ"],
                                            range=[LOWER, RAISE])),
            tooltip=[alt.Tooltip("name:N", title="ປັດໄຈ"),
                     alt.Tooltip("group:N", title="ກຸ່ມ"),
                     alt.Tooltip("share:Q", title="ສັດສ່ວນ (%)", format=".1f"),
                     alt.Tooltip("z:Q", title="z-score", format=".2f"),
                     alt.Tooltip("r2:Q", title="R² ຂອງໂມເດວ", format=".2f")],
        )
        .properties(width="container", height=max(160, 55 * len(d)))
    )


def chart_group(raising):
    g = raising.groupby("group", as_index=False)["share"].sum()
    return (
        alt.Chart(g)
        .mark_arc(innerRadius=62, outerRadius=105)
        .encode(
            theta=alt.Theta("share:Q"),
            color=alt.Color("group:N", title=None,
                            scale=alt.Scale(domain=["ສະພາບອາກາດ", "ກາຊມົນລະພິດ"],
                                            range=[LOWER, RAISE])),
            tooltip=[alt.Tooltip("group:N", title="ກຸ່ມ"),
                     alt.Tooltip("share:Q", title="ສັດສ່ວນ (%)", format=".1f")],
        )
        .properties(height=240)
    )


def chart_z(df):
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("z:Q", title="ຫ່າງຈາກຄ່າປົກກະຕິຂອງເດືອນ-ຊົ່ວໂມງ (ຈຳນວນເທົ່າຂອງ SD)"),
            y=alt.Y("name:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelFontSize=13)),
            color=alt.Color("direction:N", title="ທິດ",
                            scale=alt.Scale(domain=DIR_DOMAIN, range=DIR_RANGE)),
            tooltip=[alt.Tooltip("name:N", title="ປັດໄຈ"),
                     alt.Tooltip("pred:Q", title="ຄ່າທຳນາຍ", format=".2f"),
                     alt.Tooltip("normal:Q", title="ຄ່າປົກກະຕິ", format=".2f"),
                     alt.Tooltip("z:Q", title="z-score", format=".2f"),
                     alt.Tooltip("direction:N", title="ທິດ")],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#888", strokeDash=[4, 3]).encode(x="x:Q")
    return (bars + zero).properties(width="container", height=max(260, 45 * len(df)))


# ----------------------------------------------------------------------------
# ສ່ວນປະກອບ UI
# ----------------------------------------------------------------------------
def hero(pm25, level, color):
    st.markdown(
        f"""
        <div style="border-left:8px solid {color};padding:14px 22px;
                    background:{color}22;border-radius:4px;">
          <div style="font-size:14px;opacity:.7;">ຄ່າ PM2.5 ທີ່ປ້ອນ</div>
          <div style="font-size:54px;font-weight:700;line-height:1.1;">
            {pm25:g} <span style="font-size:18px;font-weight:400;opacity:.7;">µg/m³</span>
          </div>
          <div style="font-size:19px;font-weight:600;color:{color};">{level}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    try:
        b = get_bundle()
    except Exception as e:
        st.error(f"ໂຫຼດໂມເດວບໍ່ໄດ້: {e}")
        st.stop()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("ປ້ອນຂໍ້ມູນ")
        pm25 = st.number_input("ຄ່າ PM2.5 (µg/m³)", min_value=0.0, max_value=1000.0,
                               value=120.0, step=5.0)
        month_name = st.selectbox("ເດືອນ", MONTHS, index=0)
        month = MONTHS.index(month_name) + 1
        
        # ຮັບຄ່າເວລາ (ຊົ່ວໂມງ)
        hour = st.number_input("ເວລາ (ໂມງ)", min_value=0, max_value=23, value=8, step=1)

        st.divider()
        meta = b.get("meta", {})

    st.title("ວິເຄາະປັດໃຈທີ່ກ່ຽວຂ້ອງກັບຄ່າ PM2.5")
    st.write("ໃສ່ຄ່າ PM2.5 / ດືອນ / ເວລາ ແລ້ວໂມເດວຈະບອກວ່າ ສະພາບອາກາດ ແລະ ກາຊມົນລະພິດຕົວໃດ "
             "ຜິດປົກກະຕິໃນທິດທີ່ເຮັດໃຫ້ຝຸ່ນສູງ ພ້ອມວິທີປ້ອງກັນ.")

    level, color = pm_level(pm25)
    df = analyze(pm25, hour, month)
    raising = df[df["direction"] == "ເພີ່ມຝຸ່ນ"].sort_values("share", ascending=False)

    c1, c2 = st.columns([1, 2])
    with c1:
        hero(pm25, level, color)
    with c2:
        if raising.empty:
            st.info("ທຸກປັດໄຈຢູ່ໃນລະດັບປົກກະຕິ (|z| < 0.5) ຫຼື ຜິດປົກກະຕິໃນທິດທີ່ຊ່ວຍຫຼຸດຝຸ່ນ "
                    "ຈຶ່ງບໍ່ມີປັດໄຈທີ່ຕ້ອງແກ້ໄຂເພີ່ມ. ເບິ່ງລາຍລະອຽດໃນແຖບ \"ທຽບຄ່າປົກກະຕິ\".")
        else:
            top = raising.iloc[0]
            weather_share = raising.loc[raising["group"] == "ສະພາບອາກາດ", "share"].sum()
            gas_share = raising.loc[raising["group"] == "ກາຊມົນລະພິດ", "share"].sum()
            m1, m2, m3 = st.columns(3)
            m1.metric("ປັດໄຈເດັ່ນສຸດ", top["name"], f"{top['share']:.1f}%", delta_color="off")
            m2.metric("ສະພາບອາກາດ", f"{weather_share:.1f}%")
            m3.metric("ກາຊມົນລະພິດ", f"{gas_share:.1f}%")
            st.write(f"ໃນເດືອນ{month_name} ເວລາ {hour}:00 ໂມງ ເມື່ອ PM2.5 ເປັນ {pm25:g} µg/m³ "
                     f"ປັດໄຈທີ່ຜິດປົກກະຕິໃນທິດເພີ່ມຝຸ່ນຫຼາຍສຸດຄື **{top['name']}** "
                     f"(ປະມານ {top['share']:.0f}% ຂອງທັງໝົດ, ຜິດປົກກະຕິ{top['severity']}).")
            if pm25 <= LEVELS[0][0]:
                st.info("PM2.5 ຢູ່ໃນລະດັບດີ (≤ 50 µg/m³). ປັດໄຈຂ້າງລຸ່ມຜິດປົກກະຕິພຽງເລັກນ້ອຍ "
                        "ຄຳແນະນຳແມ່ນເພື່ອຮັກສາ ແລະ ປັບໃຫ້ດີກວ່າປົກກະຕິ ບໍ່ໄດ້ໝາຍວ່າຝຸ່ນຢູ່ໃນລະດັບອັນຕະລາຍ.")

    # ປັບປ່ຽນເຫຼືອພຽງ 3 ແທັບຫຼັກ
    tab1, tab2, tab3 = st.tabs(["ສາເຫດຫຼັກທີ່ກ່ຽວຂ້ອງ", "ທຽບຄ່າປົກກະຕິ", "ວິທີປ້ອງກັນ"])

    # ---------------- Tab 1 ----------------
    with tab1:
        if raising.empty:
            st.write("ບໍ່ມີປັດໄຈທີ່ຕ້ອງສະແດງ.")
        else:
            left, right = st.columns([3, 2])
            with left:
                st.subheader("ສັດສ່ວນຂອງປັດໄຈທີ່ຜັກດັນໃຫ້ຝຸ່ນສູງ")
                st.altair_chart(chart_share(raising))
            with right:
                st.subheader("ແບ່ງຕາມກຸ່ມ")
                st.altair_chart(chart_group(raising))
            st.caption("ສັດສ່ວນນີ້ຄືຄວາມຜິດປົກກະຕິທີ່ຖ່ວງນ້ຳໜັກດ້ວຍຜົນຊ່ວຍຂອງ PM2.5 ຕໍ່ການທຳນາຍປັດໄຈນັ້ນ "
                       "ໃນບັນດາປັດໄຈທີ່ເພີ່ມຝຸ່ນ ຜົນລວມ = 100%. ບໍ່ແມ່ນສັດສ່ວນຂອງສາເຫດທີ່ແທ້ຈິງ.")
            st.subheader("ລະດັບຄວາມຜິດປົກກະຕິຂອງແຕ່ລະປັດໄຈ")
            for _, r in raising.iterrows():
                st.markdown(f"- **{r['name']}**: ຜິດປົກກະຕິ**{r['severity']}** (z = {r['z']:+.2f}). "
                            f"{SEVERITY_NOTE[r['severity']]}")

    # ---------------- Tab 2 ----------------
    with tab2:
        st.subheader("ແຕ່ລະປັດໄຈຫ່າງຈາກຄ່າປົກກະຕິຂອງເດືອນ-ຊົ່ວໂມງເທົ່າໃດ")
        order = df.reindex(df["z"].abs().sort_values(ascending=False).index)
        st.altair_chart(chart_z(order))
        st.caption("ແດງ = ຜິດປົກກະຕິໃນທິດທີ່ສຳພັນກັບຝຸ່ນສູງ, ຟ້າ = ທິດທີ່ສຳພັນກັບຝຸ່ນຕ່ຳ, "
                   "ເທົາ = ຢູ່ໃນລະດັບປົກກະຕິ ຫຼື ທິດບໍ່ຊັດເຈນ.")

        table = pd.DataFrame({
            "ປັດໄຈ": order["name"],
            "ຄ່າທຳນາຍ": order.apply(lambda r: f"{r['pred']:.1f} {r['unit']}", axis=1),
            "ຊ່ວງ (P10-P90)": order.apply(lambda r: f"{r['low']:.1f} - {r['high']:.1f}", axis=1),
            "ຄ່າປົກກະຕິ (ເດືອນ-ຊົ່ວໂມງ)": order.apply(lambda r: f"{r['normal']:.1f} {r['unit']}", axis=1),
            "z-score": order["z"].round(2),
            "ທິດ": order["direction"],
            "R² ຂອງໂມເດວ": order["r2"].round(2),
        }).reset_index(drop=True)
        st.dataframe(table, hide_index=True)

        weak = order[order["r2"] < 0.4]
        if not weak.empty:
            st.warning("ປັດໄຈທີ່ໂມເດວທຳນາຍໄດ້ບໍ່ແມ່ນຍຳ (R² < 0.4): " + ", ".join(weak["name"]) +
                       ". ຄວນເບິ່ງເປັນຄ່າປະມານ.")

    # ---------------- Tab 3 ----------------
    with tab3:
        st.subheader("ປົກປ້ອງສຸຂະພາບ")
        advice_html = "".join(
            "<li style='margin-bottom: 8px;'>" + md_bold(tip) + "</li>"
            for tip in HEALTH_ADVICE[level])
        st.markdown(
            f"""
            <div style="border-left: 6px solid {color}; background-color: {color}15; padding: 18px; border-radius: 6px; margin-bottom: 20px;">
                <h4 style="margin-top: 0; color: {color}; margin-bottom: 12px;">ລະດັບຄຸນນະພາບອາກາດ: {level}</h4>
                <ul style="margin: 0; padding-left: 20px; line-height: 1.6; font-size: 15px;">
                    {advice_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("ວິທີປ້ອງກັນ ແລະ ແກ້ໄຂປັດໄຈທີ່ຜິດປົກກະຕິ")
        if raising.empty:
            st.write("ບໍ່ມີປັດໄຈທີ່ຜິດປົກກະຕິໃນຄັ້ງນີ້ ໃຫ້ປະຕິບັດຕາມຄຳແນະນຳດ້ານສຸຂະພາບຂ້າງເທິງ.")
        for i, (_, r) in enumerate(raising.iterrows()):
            adv = FACTOR_ADVICE[r["col"]]
            title = f"{r['name']} ({r['share']:.0f}%, ຜິດປົກກະຕິ{r['severity']})"
            with st.expander(title, expanded=(i < 3)):
                st.write(f"ຄ່າທຳນາຍ {r['pred']:.1f} {r['unit']} ທຽບກັບຄ່າປົກກະຕິ {r['normal']:.1f} {r['unit']} (z = {r['z']:+.2f}).")
                st.write(adv["why"])
                st.markdown("**" + GOAL[r["col"]].format(normal=r["normal"], unit=r["unit"]) + "**")
                a, c = st.columns(2)
                with a:
                    st.markdown("**ສ່ວນບຸກຄົນ**")
                    for t in adv["personal"]:
                        st.markdown(f"- {t}")
                with c:
                    st.markdown("**ຊຸມຊົນ / ໜ່ວຍງານ**")
                    for t in adv["community"]:
                        st.markdown(f"- {t}")


main()