# Model Deployment (Iris prediction) using FastAPI

*Scope*: Model deployment using FastAPI

*Information*: Prediction of Citizen controlled vocabulary (UC: Turkish)

## How to run

For creating/training the prediction model, see the instructions in folder ``Turkish``

We can use the Dockerfile to create an image for running our web application inside a container
```
$ docker build . -t turkish_model_api
```
And now we can test our application using Docker
```
$ docker run -p 8000:8000 turkish_model_api
```

To check your application
```
# curl -X GET http://localhost:8000/
# curl -X GET http://localhost:8000/info
# curl -X POST http://localhost:8000/predict -H "accept: application/json" -H "Content-Type: application/json" -d "{\"text\": \"Gülen\"}"
  ```


## View on http://localhost:8000/docs

If you’ve successfully reached until here, you should have your image classifier API up and running on http://0.0.0.0:8000/docs/ and should have a similar looking page!

### Main page

<p align="center">
<img src=".\images\image1.png" width = "800" alt="" align=center />
</p>

### Use /predict

<p align="center">
<img src=".\images\image2.png" width = "800" alt="" align=center />
</p>
