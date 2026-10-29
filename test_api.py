import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.database.session import init_db

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    init_db()

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] in ["healthy", "ok"]
    assert "device" in data
    assert "active_model" in data

def test_dashboard_stats():
    response = client.get("/api/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "unique_categories_count" in data
    assert "active_model_version" in data
    assert "recent_predictions" in data
    assert isinstance(data["category_distribution"], list)
    assert isinstance(data["recent_activity"], list)

def test_dataset_info_and_classes():
    # Info
    info_res = client.get("/api/dataset/info")
    assert info_res.status_code == 200
    info_data = info_res.json()
    assert info_data["class_count"] == 101
    assert "splits" in info_data
    assert "preprocessing_info" in info_data

    # Classes
    classes_res = client.get("/api/dataset/classes")
    assert classes_res.status_code == 200
    classes_data = classes_res.json()
    assert classes_data["total"] == 101
    assert "pizza" in classes_data["classes"]
    assert "hamburger" in classes_data["classes"]

def test_dataset_samples():
    response = client.get("/api/dataset/samples")
    assert response.status_code == 200
    data = response.json()
    assert len(data["samples"]) > 0
    assert "image_url" in data["samples"][0]

def test_model_info_and_evaluation():
    # Info
    info_res = client.get("/api/model/info")
    assert info_res.status_code == 200
    info_data = info_res.json()
    assert "version_name" in info_data
    assert "educational_concepts" in info_data
    assert "convolution" in info_data["educational_concepts"]

    # Evaluation
    eval_res = client.get("/api/model/evaluation")
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert "test_accuracy" in eval_data
    assert "macro_f1_score" in eval_data
    assert "per_class_metrics" in eval_data
    assert "evaluation_report_markdown" in eval_data

    # Compare
    compare_res = client.get("/api/model/compare")
    assert compare_res.status_code == 200
    compare_data = compare_res.json()
    assert len(compare_data["comparisons"]) >= 2

def test_predict_valid_and_invalid_image():
    # Invalid extension
    bad_file = io.BytesIO(b"not an image file")
    bad_res = client.post("/api/predict", files={"file": ("test.txt", bad_file, "text/plain")})
    assert bad_res.status_code == 400

    # Valid PNG image
    img = Image.new("RGB", (128, 128), color=(200, 100, 50))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    pred_res = client.post("/api/predict", files={"file": ("food_test.png", img_byte_arr, "image/png")})
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert "id" in pred_data
    assert "predicted_class" in pred_data
    assert "confidence_percentage" in pred_data
    assert len(pred_data["top_five"]) == 5
    assert pred_data["processing_time_ms"] > 0

    pred_id = pred_data["id"]

    # Retrieve history
    hist_res = client.get("/api/predictions")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert hist_data["total"] >= 1
    assert any(p["id"] == pred_id for p in hist_data["items"])

    # Detail
    detail_res = client.get(f"/api/predictions/{pred_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == pred_id

    # Delete
    del_res = client.delete(f"/api/predictions/{pred_id}")
    assert del_res.status_code == 200

def test_training_validation():
    # Bad epochs
    bad_res = client.post("/api/training/start", json={"epochs": 100, "batch_size": 16})
    assert bad_res.status_code == 422

    # Status endpoint
    status_res = client.get("/api/training/status")
    assert status_res.status_code == 200
