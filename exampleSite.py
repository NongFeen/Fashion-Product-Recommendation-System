from http.client import HTTPException
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import base64
import os
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi import Response
app = FastAPI()

# Load csv file
csv_path = 'dataset/fix_styles.csv'

# Load pre-saved features and filenames
feature_list = np.array(pickle.load(open('extracted_feature.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Function to extract features from an image
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to recommend similar images
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return distances, indices

# Function to convert image to base64 string
def image_to_base64(img_path):
    img = Image.open(img_path)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def extract_id_from_path(file_path):
    # Remove "dataset/images/" and ".jpg" from the path
    base_filename = os.path.basename(file_path)
    return base_filename.rstrip(".jpg")

# Static Images
app.mount("/image", StaticFiles(directory="dataset/images"), name="image")

# API endpoint to handle image uploads and return recommendations
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
        
    # Extract features from the uploaded image
    features = feature_extraction(file_location, model)

    # Get recommendations and distances
    distances, indices = recommend(features, feature_list)

    # Prepare the recommendations with base64 images
    recommendations = []
    # for i in range(len(indices[0])):
    #     img_path = filenames[indices[0][i]]
    #     img_base64 = image_to_base64(img_path)
    #     recommendations.append({
    #         "image_base64": img_base64,
    #         "distance": float(distances[0][i])
    #     })
    for i in range(10):
        img_id = extract_id_from_path(filenames[indices[0][i]])
        recommendations.append({
            "id": img_id,  # Use the image ID instead of the index
        })
    # Return the recommendations
    return JSONResponse(content= recommendations)

@app.get("/metadata")
async def getMetadata():
    try:
        df = pd.read_csv(csv_path, delimiter=';') 
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        json_data = df.to_dict(orient='records')
        
        json_data = [{k: (None if pd.isna(v) else v) for k, v in item.items()} for item in json_data] # don't touch this 
        # Somehow this is needed to run
        # for item loop each item 
        # for kv loop in key and value 
        # pd.isna(v) check if have value or not then replace it with 'None' as json can process with that
        return JSONResponse(content=json_data)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
# some missing value in csv make me pain   - Feen
@app.get("/Frontpage/")
async def getFrontpage(limit: int = Query(10, enum=[10, 20, 30, 50])):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path, delimiter=';')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Filter only the necessary columns
        filtered_df = df[['productDisplayName', 'price', 'id']]
        
        # Limit the number of records based on the query parameter
        limited_df = filtered_df.head(limit)
        
        # Convert to JSON-friendly format
        json_data = limited_df.to_dict(orient='records')
        json_data = [{k: (None if pd.isna(v) else v) for k, v in item.items()} for item in json_data]
        
        return JSONResponse(content=json_data)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/detail/{item_id}")
def getItemDetail(item_id: str):
    df = pd.read_csv(csv_path, delimiter=';')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    item_detail = df[df['id'] == int(item_id)]#find by id 

    if item_detail.empty:
        raise HTTPException(status_code=404, detail="Item not found")

    item_dict = item_detail.to_dict(orient='records')[0]
    item_dict = {k: (None if pd.isna(v) else v) for k, v in item_dict.items()}

    return JSONResponse(content=item_dict)

@app.get("/searchbyname") 
def get_item_detail_by_name(name: str):
    df = pd.read_csv(csv_path, delimiter=';')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    item_detail = df[df['productDisplayName'].str.contains(name, case=False, na=False)]#find by name
    if item_detail.empty:
        raise HTTPException(status_code=404, detail="No items found")

    items_list = item_detail.to_dict(orient='records')
    items_list = [{k: (None if pd.isna(v) else v) for k, v in item.items()} for item in items_list]
    #example | searchbyname?name=Turtle Check men 
    #will turn to | searchbyname?name=Turtle%20Check%20men
    return JSONResponse(content=items_list)

@app.get("/getMultiple")
def get_multiple_by_id(ids: str):
    df = pd.read_csv(csv_path, delimiter=';')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    filtered_df = df[['productDisplayName', 'price', 'id']]
    id_list = [int(i) for i in ids.split(",")]
    item_details = filtered_df[filtered_df['id'].isin(id_list)]

    if item_details.empty:
        raise HTTPException(status_code=404, detail="No items found")
    
    items_list = item_details.to_dict(orient='records')
    items_list = [{k: (None if pd.isna(v) else v) for k, v in item.items()} for item in items_list]
    #example /getMultiple?ids=9204,6842,13089
    return JSONResponse(content=items_list)

@app.get("/searchbyDetail") 
def get_item_detail_by_name(
    gender: str = None,
    masterCategory: str = None,
    subCategory: str = None,
    articleType: str = None,
    baseColour: str = None,
    season: str = None,
    year: str = None,
    usage : str = None):
    df = pd.read_csv(csv_path, delimiter=';')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    item_detail = df
    #filter each categories one by one
    if gender:
        item_detail = item_detail[item_detail['gender'].str.contains(gender, case=False, na=False)]
    if masterCategory:
        item_detail = item_detail[item_detail['masterCategory'].str.contains(masterCategory, case=False, na=False)]
    if subCategory:
        item_detail = item_detail[item_detail['subCategory'].str.contains(subCategory, case=False, na=False)]
    if articleType:
        item_detail = item_detail[item_detail['articleType'].str.contains(articleType, case=False, na=False)]
    if baseColour:
        item_detail = item_detail[item_detail['baseColour'].str.contains(baseColour, case=False, na=False)]
    if season:
        item_detail = item_detail[item_detail['season'].str.contains(season, case=False, na=False)]
    if year:
        item_detail = item_detail[item_detail['year'].astype(str).str.contains(year, case=False, na=False)]
    if usage:
        item_detail = item_detail[item_detail['usage'].str.contains(usage, case=False, na=False)]
    if item_detail.empty:
        raise HTTPException(status_code=404, detail="No items found")

    items_list = item_detail.to_dict(orient='records')
    items_list = [{k: (None if pd.isna(v) else v) for k, v in item.items()} for item in items_list]
    #example /searchbyDetail?gender=Men&masterCategory=Apparel
    return JSONResponse(content=items_list)
