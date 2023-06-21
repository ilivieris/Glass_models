from fastapi import FastAPI, HTTPException
from typing import TypeVar
from pydantic import BaseModel
import pickle
import spacy
import spacyturk

app = FastAPI(title = 'Citizen controlled vocabulary model prediction (UC: Turkish)')


class Parameters():
    def __init__( self ):
        self.model_name = "Citizen controlled vocabulary model prediction (UC: Turkish)"
        self.version    = "v1.0"
        self.author     = "NovelCore"
        self.model_path = 'checkpoint/model.pkl'
        self.spacy_model = 'tr_floret_web_md'
        self.label_encoder_path = 'checkpoint/label_encoder.sav'

        self.model = pickle.load(open(self.model_path, 'rb'))
        spacyturk.download(self.spacy_model)
        self.nlp = spacy.load(self.spacy_model)
        self.encoder = pickle.load(open(self.label_encoder_path, 'rb'))

args = Parameters()


class Inputs( BaseModel ):
    '''
        Creating a class for the attributes input to the ML model.
    '''
    text: str = ''



@app.get("/")
async def main():
    '''
    Base
    '''
    return {'msg': 'Hello World', 'status': 200,}

@app.get("/info")
async def info():
    """Returns model information, version and author"""
    result = dict()

    result['name']    = args.model_name
    result['version'] = args.version
    result['author']  = args.author
    result['status']  = 200

    return result

@app.post('/predict' )
async def get_model_response(data: Inputs):
    '''
    Returns models prediction on input data

    Parameters
    ----------
    text: Input text for classification
    ''' 
   
    # Load model
    #
    try:
        # Get embeddings
        embeddings = args.nlp(data.dict()['text']).vector
        # Get model's prediction
        pred = args.model.predict(embeddings.reshape(1,-1))
        pred = args.encoder.inverse_transform(pred)[0]

    except Exception as e:
        raise HTTPException(
            status_code = 404, detail = '[ERROR] ' + str(e).split('] ')[-1].strip()
        )  
    

    return {'Prediction':  pred, 'status': 200}

