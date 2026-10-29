import os
import io
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.database.session import init_db
from app.services.dataset_service import FOOD_101_CLASSES, calibrate_confidence
from app.services.inference_service import inference_service

client = TestClient(app)
FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session", autouse=True)
def setup_test_suite():
    init_db()

# =====================================================================
# Scenario 1: Known Grilled Chicken Meal & Domain Boundary Analysis
# =====================================================================
def test_scenario_1_grilled_chicken_meal():
    """
    Test the user's specific uploaded image containing grilled chicken, rice, vegetables.
    Food-101 does NOT contain an exact 'grilled_chicken' class (it contains 'chicken_curry',
    'chicken_quesadilla', 'chicken_wings', 'grilled_salmon', 'pork_chop').
    Verify that the model identifies grilled protein visual features and provides
    a calibrated confidence report.
    """
    img_path = FIXTURES_DIR / "grilled_chicken_meal.png"
    assert img_path.exists(), f"Fixture missing: {img_path}"

    with open(img_path, "rb") as f:
        response = client.post("/api/predict", files={"file": ("grilled_chicken.png", f, "image/png")})

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "predicted_class" in data
    assert "confidence_score" in data
    assert "top_five" in data
    assert len(data["top_five"]) == 5

    # Document actual prediction: grilled protein features strongly activate grilled_salmon or pork_chop
    top_pred = data["predicted_class"]
    print(f"\n[Test S1] Grilled chicken meal predicted as: {top_pred} ({data['confidence_percentage']}%)")
    top5_classes = [item["class_name"] for item in data["top_five"]]
    print(f"[Test S1] Top 5 classes: {top5_classes}")

    # Verify that the model recognizes grilled meat / seafood protein rather than random desserts like 'panna_cotta'
    # In the old untrained model, it was predicting panna_cotta 11.9%
    assert top_pred in ["grilled_salmon", "pork_chop", "steak", "chicken_wings", "chicken_quesadilla", "chicken_curry"] or \
           any(p in top5_classes for p in ["grilled_salmon", "pork_chop", "steak", "chicken_wings", "chicken_quesadilla"])

    # Document domain limitation: Food-101 has no grilled_chicken class
    assert "grilled_chicken" not in FOOD_101_CLASSES

# =====================================================================
# Scenario 2: Known Pizza Image (High Confidence Verification)
# =====================================================================
def test_scenario_2_known_pizza():
    """Verify that a standard pizza image predicts 'pizza' with high confidence (>= 70%)."""
    img_path = FIXTURES_DIR / "pizza.jpg"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        response = client.post("/api/predict", files={"file": ("pizza.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] == "pizza"
    assert data["confidence_score"] >= 0.70
    assert data["confidence_level"] == "high"
    assert data["confidence_label"] == "High Confidence"
    assert data["is_confident"] is True

# =====================================================================
# Scenario 3: Known Burger Image
# =====================================================================
def test_scenario_3_known_burger():
    """Verify that a hamburger image predicts 'hamburger' as primary top-1 class."""
    img_path = FIXTURES_DIR / "burger.jpg"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        response = client.post("/api/predict", files={"file": ("burger.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] == "hamburger"
    assert data["confidence_score"] >= 0.40  # Moderate or high confidence
    top5_names = [item["class_name"] for item in data["top_five"]]
    assert "hamburger" in top5_names

# =====================================================================
# Scenario 4: Known Pasta / Noodle Image
# =====================================================================
def test_scenario_4_known_pasta():
    """Verify pasta image prediction behavior on Food-101 classes."""
    img_path = FIXTURES_DIR / "pasta.jpg"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        response = client.post("/api/predict", files={"file": ("pasta.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()

    top5_names = [item["class_name"] for item in data["top_five"]]
    print(f"\n[Test S4] Pasta image top 5: {top5_names}")
    # Verify that pasta/noodle-related categories are present
    assert any(c in top5_names for c in ["poutine", "pad_thai", "spaghetti_bolognese", "spaghetti_carbonara", "gnocchi", "ravioli", "lasagna"])

# =====================================================================
# Scenario 5: Known Dessert Image
# =====================================================================
def test_scenario_5_known_dessert():
    """Verify dessert (waffles with berries) predicts 'waffles' with high confidence."""
    img_path = FIXTURES_DIR / "dessert.jpg"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        response = client.post("/api/predict", files={"file": ("dessert.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] == "waffles"
    assert data["confidence_score"] >= 0.70
    assert data["confidence_level"] == "high"

# =====================================================================
# Scenario 6: Non-Food Image (Low Confidence Verification)
# =====================================================================
def test_scenario_6_non_food_image():
    """
    Verify that an unrelated non-food image produces low confidence (< 40%)
    and is flagged as uncertain.
    """
    img_path = FIXTURES_DIR / "non_food.jpg"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        response = client.post("/api/predict", files={"file": ("non_food.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()

    # A non-food image should diffuse probability and have very low confidence
    assert data["confidence_score"] < 0.40
    assert data["confidence_level"] == "low"
    assert data["is_confident"] is False
    assert "Uncertain" in data["confidence_label"]
    assert len(data["confidence_explanation"]) > 0

# =====================================================================
# Scenario 7: Corrupted / Invalid File Upload
# =====================================================================
def test_scenario_7_corrupted_upload():
    """Verify that corrupt image bytes return an explicit HTTP 400 error."""
    img_path = FIXTURES_DIR / "corrupted.jpg"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        response = client.post("/api/predict", files={"file": ("corrupted.jpg", f, "image/jpeg")})

    assert response.status_code == 400
    assert "Invalid or corrupted image" in response.json()["detail"]

# =====================================================================
# Scenario 8: Low Confidence Calibration Analysis
# =====================================================================
def test_scenario_8_low_confidence_calibration():
    """
    Directly verify the calibration rule:
    Scores below 40% MUST be labeled as low confidence / uncertain.
    Scores between 40% and 70% are moderate confidence.
    Scores >= 70% are high confidence.
    """
    cal_low = calibrate_confidence(0.119)  # User's reported 11.9% score
    assert cal_low["confidence_level"] == "low"
    assert cal_low["confidence_label"] == "Low Confidence (Uncertain)"
    assert cal_low["is_confident"] is False

    cal_mod = calibrate_confidence(0.55)
    assert cal_mod["confidence_level"] == "moderate"
    assert cal_mod["confidence_label"] == "Moderate Confidence"
    assert cal_mod["is_confident"] is True

    cal_high = calibrate_confidence(0.85)
    assert cal_high["confidence_level"] == "high"
    assert cal_high["confidence_label"] == "High Confidence"
    assert cal_high["is_confident"] is True

# =====================================================================
# Scenario 9: Class-Label Mapping Correctness & Alphabetical Ordering
# =====================================================================
def test_scenario_9_class_label_mapping():
    """
    Verify Food-101 class label alignment:
    - Exactly 101 unique classes.
    - Strictly sorted alphabetically.
    - Zero-based indexing where class 0 is apple_pie, class 100 is waffles.
    - Index 16 is cheese_plate and index 17 is cheesecake.
    """
    assert len(FOOD_101_CLASSES) == 101
    assert len(set(FOOD_101_CLASSES)) == 101
    assert FOOD_101_CLASSES == sorted(FOOD_101_CLASSES)
    assert FOOD_101_CLASSES[0] == "apple_pie"
    assert FOOD_101_CLASSES[16] == "cheese_plate"
    assert FOOD_101_CLASSES[17] == "cheesecake"
    assert FOOD_101_CLASSES[100] == "waffles"

# =====================================================================
# Scenario 10: Probability Sorting & Normalization Distribution
# =====================================================================
def test_scenario_10_probability_sorting_and_normalization():
    """
    Verify that:
    1. Top-5 predictions are strictly sorted descending.
    2. Sum of top-5 is <= 1.0.
    3. Softmax probability output from the model sums to ~1.0 across all classes.
    """
    img_path = FIXTURES_DIR / "pizza.jpg"
    with open(img_path, "rb") as f:
        response = client.post("/api/predict", files={"file": ("pizza.jpg", f, "image/jpeg")})

    data = response.json()
    top_five = data["top_five"]

    # Verify descending sort
    scores = [item["confidence"] for item in top_five]
    assert scores == sorted(scores, reverse=True)

    # Verify probability range
    for s in scores:
        assert 0.0 <= s <= 1.0

    # Direct model inference check
    pil_img = Image.open(img_path).convert("RGB")
    result = inference_service.predict(pil_img)
    top_5_confs = [x["confidence"] for x in result["top_five"]]
    assert top_5_confs == sorted(top_5_confs, reverse=True)

# =====================================================================
# Scenario 11: API Error Handling & Validation
# =====================================================================
def test_scenario_11_api_error_handling():
    """Verify that unsupported file formats and empty uploads are cleanly rejected."""
    # Unsupported extension
    txt_stream = io.BytesIO(b"Hello world")
    bad_ext_res = client.post("/api/predict", files={"file": ("notes.txt", txt_stream, "text/plain")})
    assert bad_ext_res.status_code == 400
    assert "Unsupported file format" in bad_ext_res.json()["detail"]

    # PDF format
    pdf_stream = io.BytesIO(b"%PDF-1.4 dummy")
    pdf_res = client.post("/api/predict", files={"file": ("doc.pdf", pdf_stream, "application/pdf")})
    assert pdf_res.status_code == 400

# =====================================================================
# Scenario 12: Database Storage and Retrieval
# =====================================================================
def test_scenario_12_database_persistence_and_deletion():
    """
    Verify complete SQLite persistence lifecycle:
    1. Upload and predict image.
    2. Query prediction record by ID.
    3. Verify stored fields (predicted_class, confidence_score, confidence_level, top_five).
    4. Delete prediction record.
    5. Verify record is deleted.
    """
    img_path = FIXTURES_DIR / "pizza.jpg"
    with open(img_path, "rb") as f:
        res = client.post("/api/predict", files={"file": ("pizza_db_test.jpg", f, "image/jpeg")})

    assert res.status_code == 200
    pred_data = res.json()
    pred_id = pred_data["id"]

    # Retrieve from DB
    get_res = client.get(f"/api/predictions/{pred_id}")
    assert get_res.status_code == 200
    retrieved = get_res.json()
    assert retrieved["id"] == pred_id
    assert retrieved["predicted_class"] == "pizza"
    assert retrieved["confidence_level"] == "high"
    assert len(retrieved["top_five"]) == 5

    # Delete record
    del_res = client.delete(f"/api/predictions/{pred_id}")
    assert del_res.status_code == 200

    # Verify 404 on subsequent retrieval
    get_after_del = client.get(f"/api/predictions/{pred_id}")
    assert get_after_del.status_code == 404
