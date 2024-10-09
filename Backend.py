from http.client import HTTPException
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import base64
import os
from typing import Annotated
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi import Response, Form
from pathlib import Path
from huggingface_cloth_segmentation import load_seg_model, generate_mask
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

model2 = load_seg_model('huggingface_cloth_segmentation/model/cloth_segm.pth')


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

@app.get("/images/{item_id}")
async def get_image(item_id: str): 
    directory = "dataset/images"
    image_path = Path(directory) / f"{item_id}.jpg" 
    if image_path.exists():
        image_bytes = image_path.read_bytes() 
        return Response(content=image_bytes, media_type="image/png")
    else:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    
# API endpoint to handle image uploads and return recommendations
@app.post("/upload")
async def upload_image(file: UploadFile = File(...), selection: str =Form(...)):
    # Save the uploaded image temporarily
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    # Apply mask using segmentation model
    output_filenames = generate_mask(file_location, model2)
    if len(output_filenames) == 0:
        output_filenames.append(file_location)

    # Extract features from the uploaded image
    features_list = []
    for file in output_filenames:
        features = feature_extraction(file, model)
        features_list.append(features)

    # Load CSV for metadata filtering
    df = pd.read_csv(csv_path, delimiter=';')
    
    # Filter based on the "masterCategory" selection
    if selection:
        df_filtered = df[df['masterCategory'].str.contains(selection, case=False, na=False)]
        if df_filtered.empty:
            return JSONResponse(content={"error": "No items found in the selected category"}, status_code=404)

        # Filter filenames and features based on the filtered dataframe
        valid_ids = df_filtered['id'].astype(str).tolist()
        filtered_filenames = [filenames[i] for i in range(len(filenames)) if extract_id_from_path(filenames[i]) in valid_ids]
        filtered_feature_list = [feature_list[i] for i in range(len(filenames)) if extract_id_from_path(filenames[i]) in valid_ids]
    else:
        filtered_filenames = filenames
        filtered_feature_list = feature_list

    indices_list = []
    distances_list = []

    for features in features_list:
        distances, indices = recommend(features, filtered_feature_list)
        indices_list.append(indices)
        distances_list.append(distances)

    recommendations = []

    num = len(indices_list)
    for i in range(10):
        img_id = extract_id_from_path(filtered_filenames[indices_list[i%num][0][i//num]])
        recommendations.append({
            "id": img_id,  # Use the image ID instead of the index
        })

    # Return the recommendations as JSON
    return JSONResponse(content=recommendations)


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
async def getFrontpage(page: int):
    try:
        # Read the CSV file
        number_per_page = 32
        start_index = (page - 1) * number_per_page  # Starting index for the page
        end_index = start_index + number_per_page
        
        df = pd.read_csv(csv_path, delimiter=';')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Filter only the necessary columns
        # filtered_df = df[['productDisplayName', 'price', 'id']]
        filtered_df = df
        
        # Limit the number of records based on the query parameter
        limited_df = filtered_df.iloc[start_index:end_index]
        
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
    item_detail = df[df['productDisplayName'].str.contains(name, case=False, na=False)]#find by nameง
    start_index = 0
    end_index = 100
    limited_df = item_detail.iloc[start_index:end_index]
    
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
    filtered_df = df
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
    usage : str = None,
    name: str = None,
    min_price: float = None, 
    max_price: float = None 
    ):
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
    if name:  
        item_detail = item_detail[item_detail['productDisplayName'].str.contains(name, case=False, na=False)]
    if min_price is not None:
        item_detail = item_detail[item_detail['price'] >= min_price]
    if max_price is not None:
        item_detail = item_detail[item_detail['price'] <= max_price]
    if item_detail.empty:
        raise HTTPException(status_code=404, detail="No items found")
    start_index = 0
    end_index = 100
    item_detail = item_detail.iloc[start_index:end_index]
    items_list = item_detail.to_dict(orient='records')
    items_list = [{k: (None if pd.isna(v) else v) for k, v in item.items()} for item in items_list]
    #example /searchbyDetail?gender=Men&masterCategory=Apparel
    return JSONResponse(content=items_list)
