from medcalc.bmi_calculator import bmi_calculator_explanation
from medcalc.ideal_body_weight import ibw_explanation
from medcalc.adjusted_body_weight import abw_explanation
from medcalc.creatinine_clearance import generate_cockcroft_gault_explanation

weight_value = 115
weight_unit = "kg"
height_value = 198
height_unit = "cm"
sex = "Male"
age_value = 43
age_unit = "years"
creatinine_value = 691.6
creatinine_unit = "μmol/L"

bmi_input = {
    "weight": (weight_value, weight_unit),
    "height": (height_value, height_unit),
}
bmi_result = bmi_calculator_explanation(bmi_input)["Answer"]
bmi = float(bmi_result)

ibw_input = {
    "height": (height_value, height_unit),
    "sex": sex,
    "age": int(age_value),
}
ibw_result = ibw_explanation(ibw_input)["Answer"]
ibw = float(ibw_result)

abw = None

if bmi < 18.5:
    # Underweight, use actual body weight
    abw_value = weight_value
elif 18.5 <= bmi < 25:
    # Normal weight: ABW = min(actual, IBW)
    abw_value = min(weight_value, ibw)
else:
    # Overweight/Obese: Use adjusted body weight
    abw_input = {
        "weight": (weight_value, weight_unit),
        "height": (height_value, height_unit),
        "sex": sex,
        "age": int(age_value),
    }
    abw_result = abw_explanation(abw_input)["Answer"]
    abw_value = float(abw_result)

creatinine_clearance_input = {
    "weight": (abw_value, weight_unit),
    "height": (height_value, height_unit),
    "sex": sex,
    "creatinine": (creatinine_value, creatinine_unit),
    "age": (age_value, age_unit),
}
result = generate_cockcroft_gault_explanation(creatinine_clearance_input)["Answer"]
print(result)