from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import uvicorn
from pydantic import field_validator, confloat, BaseModel, Field

app = FastAPI(
    title="Deploy breast cancer model - best model",
    version="0.0.5"
)

# ------------------------------------------------------------
# LOAD THE AI MODEL
# ------------------------------------------------------------
model = joblib.load("model/logistic_regression_model.pkl")


class BreastCancerFeatures(BaseModel):
    radius_mean: int = Field(title="Radius mean")
    texture_mean: confloat(gt=0, lt=100)
    perimeter_mean: confloat(gt=0, lt=300)
    area_mean: confloat(gt=0, lt=3000)
    smoothness_mean: confloat(gt=0, lt=2)
    compactness_mean: confloat(gt=0, lt=2)
    concavity_mean: confloat(gt=0, lt=2)
    concave_points_mean: confloat(gt=0, lt=2)
    symmetry_mean: confloat(gt=0, lt=2)
    fractal_dimension_mean: confloat(gt=0, lt=2)
    radius_se: confloat(gt=0, lt=10)
    texture_se: confloat(gt=0, lt=10)
    perimeter_se: confloat(gt=0, lt=50)
    area_se: confloat(gt=0, lt=1000)
    smoothness_se: confloat(gt=0, lt=2)
    compactness_se: confloat(gt=0, lt=2)
    concavity_se: confloat(gt=0, lt=2)
    concave_points_se: confloat(gt=0, lt=2)
    symmetry_se: confloat(gt=0, lt=2)
    fractal_dimension_se: confloat(gt=0, lt=2)
    radius_worst: confloat(gt=0, lt=100)
    texture_worst: confloat(gt=0, lt=100)
    perimeter_worst: confloat(gt=0, lt=300)
    area_worst: confloat(gt=0, lt=4000)
    smoothness_worst: confloat(gt=0, lt=2)
    compactness_worst: confloat(gt=0, lt=2)
    concavity_worst: confloat(gt=0, lt=2)
    concave_points_worst: confloat(gt=0, lt=2)
    symmetry_worst: confloat(gt=0, lt=2)
    fractal_dimension_worst: confloat(gt=0, lt=2)

    @field_validator("radius_mean")
    def validate_radius_mean(cls, value):
        # if isinstance(value, int):
        #     return value

        if value < 80:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Radius mean must be greater than or equal to 80."
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Radius mean is invalid."
        )


@app.post("/api/v2/predict-breast-cancer", tags=["breast-cancer"])
async def predict(
        features: BreastCancerFeatures
):
    dictionary = {
        'radius_mean': features.radius_mean,
        'texture_mean': features.texture_mean,
        'perimeter_mean': features.perimeter_mean,
        'area_mean': features.area_mean,
        'smoothness_mean': features.smoothness_mean,
        'compactness_mean': features.compactness_mean,
        'concavity_mean': features.concavity_mean,
        'concave points_mean': features.concave_points_mean,
        'symmetry_mean': features.symmetry_mean,
        'fractal_dimension_mean': features.fractal_dimension_mean,
        'radius_se': features.radius_se,
        'texture_se': features.texture_se,
        'perimeter_se': features.perimeter_se,
        'area_se': features.area_se,
        'smoothness_se': features.smoothness_se,
        'compactness_se': features.compactness_se,
        'concavity_se': features.concavity_se,
        'concave points_se': features.concave_points_se,
        'symmetry_se': features.symmetry_se,
        'fractal_dimension_se': features.fractal_dimension_se,
        'radius_worst': features.radius_worst,
        'texture_worst': features.texture_worst,
        'perimeter_worst': features.perimeter_worst,
        'area_worst': features.area_worst,
        'smoothness_worst': features.smoothness_worst,
        'compactness_worst': features.compactness_worst,
        'concavity_worst': features.concavity_worst,
        'concave points_worst': features.concave_points_worst,
        'symmetry_worst': features.symmetry_worst,
        'fractal_dimension_worst': features.fractal_dimension_worst
    }

    try:
        df = pd.DataFrame(dictionary, index=[0])
        prediction = model.predict(df)[0]

        if prediction == 1:
            prediction = "Es una tumoración maligna"
        else:
            prediction = "Es una tumoración benigna"

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=str(prediction)
        )
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )


if __name__ == "__main__":
    uvicorn.run(
        "endpoints:app",
        host="0.0.0.0",
        port=8000
    )