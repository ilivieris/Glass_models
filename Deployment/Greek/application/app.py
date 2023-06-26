from fastapi import FastAPI, HTTPException
from typing import TypeVar
from pydantic import BaseModel
import pickle
import spacy
from application.utils import hard_rules

app = FastAPI(title = 'Citizen controlled vocabulary model prediction (UC: Greek)')

Model = TypeVar('Model')
def load_spacy_model(model_name: str) -> Model:
    """
    Function which loads or downloads, the required nlp model, while disabling a list of unnecessary components.
    
    Parameters
    ----------
    model_name: path to spaCy model (str).

    Returns
    -------
    nlp: the spacy nlp object (Model)
    """
    try:
        nlp = spacy.load(model_name)
    except OSError:
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    return nlp



class Parameters():
    def __init__( self ):
        self.model_name = "Citizen controlled vocabulary model prediction (UC: Greek)"
        self.version    = "v1.0"
        self.author     = "NovelCore"
        self.model_path = 'Model/model.pkl'
        self.spacy_model = 'el_core_news_sm'
        self.label_encoder_path = 'Model/Label_encoder.pkl'

        self.model = pickle.load(open(self.model_path, 'rb'))
        self.nlp = load_spacy_model(self.spacy_model)
        self.encoder = pickle.load(open(self.label_encoder_path, 'rb'))

args = Parameters()


class Inputs( BaseModel ):
    '''
        Creating a class for the attributes input to the ML model.
    '''
    value: str = ''
    evidence_type: str = ''



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
        value = data.dict()['value']
        evidence_type = data.dict()['evidence_type']

        pred = hard_rules(value, evidence_type)

        if pred is None:
            # Get embeddings
            embeddings = args.nlp(value).vector
            # Get model's prediction
            pred = args.model.predict(embeddings.reshape(1,-1))
            pred = args.encoder.inverse_transform(pred)[0]

    except Exception as e:
        raise HTTPException(
            status_code = 404, detail = '[ERROR] ' + str(e).split('] ')[-1].strip()
        )  
    

    return {'Prediction':  pred, 'status': 200}

