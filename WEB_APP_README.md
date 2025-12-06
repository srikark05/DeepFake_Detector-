# DeepFake Detection Web Application

A modern web interface for the TDA-based DeepFake detection model, built with FastAPI and HTML.

## Features

- 🎨 Beautiful, modern UI with drag-and-drop file upload
- 🚀 Fast API backend using FastAPI
- 📊 Real-time prediction results with confidence scores
- 🖼️ Image preview before analysis
- 📱 Responsive design

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Before using the web app, you need to train and save the model:

```bash
python train_model.py
```

This will:
- Train the TDA-based SVM classifier on your dataset
- Save the trained model to `trained_model.pkl`
- Display training metrics

**Note:** This may take a while depending on your dataset size. The default in `pipeline.py` limits to 50 images per class for testing. You can modify `max_images` in `pipeline.py` line 298 to use the full dataset.

### 3. Start the Web Server

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload
```

The server will start at `http://localhost:8000`

### 4. Open in Browser

Navigate to `http://localhost:8000` in your web browser.

## Usage

1. **Upload Image**: Drag and drop an image onto the upload area, or click to browse
2. **Wait for Analysis**: The image will be processed (this may take 10-30 seconds)
3. **View Results**: See the prediction (Real/Fake) with confidence scores and probabilities

## API Endpoints

### `GET /`
Serves the main HTML interface.

### `POST /api/predict`
Upload an image and get prediction results.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "prediction": "Real" | "Fake",
  "confidence": 0.95,
  "probabilities": {
    "fake": 0.05,
    "real": 0.95
  }
}
```

### `GET /api/health`
Check if the API is running and if the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Project Structure

```
.
├── app.py                 # FastAPI backend
├── train_model.py         # Model training script
├── pipeline.py            # Core TDA pipeline
├── trained_model.pkl      # Saved model (created after training)
├── templates/
│   └── index.html         # Frontend HTML
└── static/                # Static files (if needed)
```

## Troubleshooting

### "Model not found" Error

Make sure you've run `train_model.py` first to create the `trained_model.pkl` file.

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Port Already in Use

If port 8000 is already in use, you can change it in `app.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8080)  # Change to any available port
```

### Slow Predictions

TDA feature extraction is computationally intensive. For faster predictions:
- Reduce image size in `app.py` (currently 128x128)
- Reduce point cloud subsample size in `pipeline.py` (currently 5000)

## Development

### Running in Development Mode

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-reload on code changes.

### Customizing the UI

Edit `templates/index.html` to customize the frontend appearance and behavior.

### Modifying the API

Edit `app.py` to add new endpoints or modify existing ones.

## Notes

- The model processes images at 128x128 resolution for efficiency
- Feature extraction may take 10-30 seconds per image
- Supported image formats: PNG, JPG, JPEG, etc. (any format supported by PIL)

