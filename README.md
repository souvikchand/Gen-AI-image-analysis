# 📱 Nano AI Image Analyzer

this is a Gen-AI project. that utilizes various AI models to analyze a picture.

### steps
1. create your virtual environment (venv) [optional]
    > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    > .\venv\Scripts\Activate.ps1  #powershell
    > venv\Scripts\activate.bat    #cmd

1. install all modules under `requirements.txt` using -->
    > pip install -r requirements.txt

2. run the app using
    > streamlit run app.py
    make sure to check `.streamlit/config.toml` for your streamlit settings


## features offered

<h3>🔍 Detect Objects</h3> 
    detects person, car, watch, TV etc <br/>
    creates a bounded box around it <br />
    provides a table with rows representing object type, box coordinates, score

<h3>📝 Describe Image</h3>
    generates a very small caption 

<h3>📖 Generate Story</h3>
    creates a story based on caption 

<h3>💬 Chat system</h3>
    ask about image 


## models used 🤖💻
why don't you read the `apps.py`? 😁

## file structure 
```
|-- .streamlit
|    |-- config.toml  --> streamlit configuration
|
|-- app.py   --> main app
|-- functions.py --> functions used in the app. (it is not a module)
|-- instruction.txt  --> draft of README.md
|-- requirements.txt --> all the pakages under venv. generated using `pip freeze`
|-- type2.py \
|-- type3.py  --> some extra template i used while developing
|-- type4.py /
```

## link of streamlit cloud
[will be added](#)