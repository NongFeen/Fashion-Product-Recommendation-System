# Fashion-Product-Recommendation-System
AI Project making Fashion Product Recommendation System Using Resnet 50 website with NextJS,React and Flask <br />
Frontend code  : https://github.com/mickie2k/Fashion-Recommendation-System-Frontend-CS-KMITL <br />
Reference
[Fashion Product Recommendation System Using Resnet 50](https://medium.com/@sharma.tanish096/fashion-product-recommendation-system-using-resnet-50-5ea5406c8f2c) <br />

Dataset must be in "dataset" file, inside must have "images" file
./dataset/images <br/>
[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

Clone huggingface_Cloth_segmentation and addd __init__.py in to the folder
```
git clone https://github.com/wildoctopus/huggingface-cloth-segmentation.git
pip install -r requirements.txt
```

Using python venv 
``` 
# Windows command prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS and Linux
source .venv/bin/activate
```
Running API using
```
uvicorn backend:app --reload
ngrok http http://localhost:8000
```
Testing POST api with img with POST method will recive id as json 
```
curl -X POST "http://localhost:8000/upload/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@pathto/image.png"
```
