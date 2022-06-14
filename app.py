from fileinput import filename
import pandas as pd
import scipy.stats as st
import numpy as np
import statsmodels.api as sm
from flask import Flask, render_template,request
from flask import Flask, render_template,request
from flask_mysqldb import MySQL
from email.message import EmailMessage
#from bs4 import BeautifulSoup
import smtplib
import ard
import datetime
import csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
import os

patient={}
sex=0
currentSmoker=0
cigsPerDay=0
BPMeds=0
prevalentStroke=0
prevalentHyp=0
diabetes=0
totChol=0
sysBP=0
diaBP=0
BMI=0.0
pulseRate=0
id=0
glucose=0
m=""
app = Flask(__name__,static_folder='static')
app.config['MYSQL_HOST']='db1.cptkkmu0skgi.ap-south-1.rds.amazonaws.com'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']='Astr0boi#00875'
app.config['MYSQL_DB']='heartpred'
mysql=MySQL(app)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/register',methods=['POST'])
def signup():
    if request.method=='POST':
        global id
        name=str(request.form['Uname'])
        password=request.form['Password']
        cur=mysql.connection.cursor()
        cur.execute("INSERT INTO user(USER_NAME,PASSWORD) values(%s,%s)",(name,password))
        cur.connection.commit()
        cur.execute("SELECT * FROM user where USER_NAME=%s and PASSWORD=%s",(name,password))
        result=cur.fetchall()
        for i in result:
            id=i[0]
        cur.close();
        return render_template('oldindex.html')
@app.route('/main',methods=['POST'])
def main():
    return render_template('home.html')
@app.route('/Home',methods=['POST'])
def home():
    global id
    if request.method=='POST':
        name=request.form['Name']
        password=request.form['Password']
        cur=mysql.connection.cursor()
        cur.execute("SELECT * FROM user where USER_NAME=%s",[name])
        result=cur.fetchall()
        for i in result:
            if i[2]==password:
                cur.execute("SELECT USER_ID FROM user WHERE USER_NAME=%s",[name])
                id=cur.fetchall()
                return render_template('home.html')
            else:
                return render_template('index.html')
@app.route('/update',methods=['GET'])
def update():
    if request.method=="GET":
        return render_template('oldindex1.html')
@app.route('/Confirm',methods=['POST'])
def confirm():
    global id
    sex=int(request.form['sex'])
    age=int(request.form['age'])
    currentSmoker = int(request.form['currentSmoker'])
    cigsPerDay = int(request.form['cigsPerDay'])
    BPMeds = int(request.form['BPMeds'])
    prevalentStroke = int(request.form['prevalentStroke'])
    prevalentHyp = int(request.form['prevalentHyp'])
    diabetes = int(request.form['diabetes'])
    totChol = int(request.form['totChol'])
    sysBP = int(request.form['sysBP'])
    diaBP = int(request.form['diaBP'])
    BMI = float(request.form['BMI'])
    glucose = int(request.form['glucose'])
    pm=request.form['pmail']
    dm=request.form['dmail']
    cur=mysql.connection.cursor()
    cur.execute("UPDATE user_details set sex=%s,age=%s,smoke=%s,cigsperday=%s,BPmed=%s,Stroke=%s,hypertension=%s,diabetes=%s,cholostrol=%s,sysBP=%s,diaBP=%s,BMI=%s,glucose=%s,p_email=%s,d_email=%s where USER_ID=%s",(sex,age,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,glucose,pm,dm,id))
    cur.connection.commit()
    cur.close()
    return render_template('home.html')
@app.route('/Measure',methods=['GET'])
def link():
    if request.method=="GET":
        return render_template('temp1.html')
# @app.route('/print',methods=['GET'])
# def printresult():
#     if request.method=="GET":
#         cur=mysql.connection.cursor()
#         cur.execute("SELECT * FROM user_result where USER_ID=%s",[id])
#         result=cur.fetchall()
#         fp = open('D:\\heartdemo\\heart-main\\Geeks1.csv', 'w')
#         myFile = csv.writer(fp)
#         pd.set_option('display.width', 1000)
#         myFile.writerows(result)  
#         fp.close()
#         df = pd.read_csv('Geeks1.csv')
#         df.to_csv("Geeks1.csv",header=['USER_ID','HEART_RATE','CIGS PER DAY','SYSBP','DIABP','CHOLOSTROL','BMI','RISK FACTOR','DATE'],index=False)
#         return render_template('home.html')


@app.route('/heartRate')
def heartRate():
    global id
    global pulseRate
    pulseRate=int(ard.heart_control())
    return render_template('temp.html',heartRate=pulseRate)
@app.route('/history',methods=['GET'])
def history():
    global id
    cur=mysql.connection.cursor()
    cur.execute("select * from user_result where USER_ID=%s",[id])
    result=cur.fetchall()
    return render_template('history.html',result=result)
@app.route('/predic')
def predic():
    global id
    global pulseRate
    cur=mysql.connection.cursor()
    cur.execute("SELECT * FROM user_details where USER_ID=%s",[id])
    result=cur.fetchall()
    for i in result:
        age=i[1]
        cigs=i[3]
        cholo=i[8]
        sys=i[9]
        dia=i[10]
        B=i[11]
        glucose=i[12]
        pm=i[13]
        dm=i[14]
    train=pd.read_csv('framingham.csv')
    train.dropna(axis=0,inplace=True)
    from statsmodels.tools import add_constant as add_constant
    train_constant = add_constant(train)
    train_constant.head()
    st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
    cols=train_constant.columns[:-1]
    model=sm.Logit(train.TenYearCHD,train_constant[cols])
    result=model.fit()
    result.summary()
    import sklearn
    new_features=train[['age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose','TenYearCHD']]
    x=new_features.iloc[:,:-1]
    y=new_features.iloc[:,-1]
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
    from sklearn.linear_model import LogisticRegression
    logreg=LogisticRegression()
    logreg.fit(x_train,y_train)
    y_pred=logreg.predict(x_test)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(logreg, x_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy: %.2f' % (mean(scores)))
    x=np.array([age,cigs,cholo,sys,dia,B,pulseRate,glucose]).reshape(1,-1)
    prediction=logreg.predict(x)     
    server=smtplib.SMTP_SSL('smtp.gmail.com',465)
    server.login("shreeshyleshronaldo@gmail.com","reyilcphxxjlvdbq")
    datetime_object = datetime.datetime.now()
    cur=mysql.connection.cursor()
    cur.execute("INSERT into user_result values(%s,%s,%s,%s,%s,%s,%s,%s,%s)",(id,pulseRate,cigs,sys,dia,cholo,B,prediction[0],datetime_object))
    cur.connection.commit()
    cur.close()
    if prediction==0:
        msgg=EmailMessage()
        msgg['From']='shreeshyleshronaldo@gmail.com'
        m="ʏᴏᴜ ʜᴀᴠᴇ no ʀɪꜱᴋ ᴏꜰ ɢᴇᴛᴛɪɴɢ ʜᴇᴀʀᴛ ᴅɪꜱᴇᴀꜱᴇ. To be on safer side consult a doctor."
        msgg['Subject'] = m
        cur=mysql.connection.cursor()
        cur.execute("SELECT * FROM user_result where USER_ID=%s",[id])
        result=cur.fetchall()
        fp = open('D:\\heartdemo\\heart-main\\report1.csv', 'w')
        myFile = csv.writer(fp)
        pd.set_option('display.width', 1000)
        myFile.writerows(result)
        fp.close()
        msgg['To']=pm
        server.send_message(msgg)
        del msgg['To']
        del msgg['Subject']
        return render_template("negative.html")
        
    else:
        msg=EmailMessage()
        msg['From']='shreeshyleshronaldo@gmail.com'
        m="ʏᴏᴜ ʜᴀᴠᴇ ʀɪꜱᴋ ᴏꜰ ɢᴇᴛᴛɪɴɢ ʜᴇᴀʀᴛ ᴅɪꜱᴇᴀꜱᴇ. ᴋɪɴᴅʟʏ ᴄᴏɴꜱᴜʟᴛ ᴀ ᴅᴏᴄᴛᴏʀ"
        msg['Subject'] = m
        cur=mysql.connection.cursor()
        cur.execute("SELECT * FROM user_result where USER_ID=%s",[id])
        result=cur.fetchall()
        fp = open('D:\\heartdemo\\heart-main\\report1.csv', 'w')
        myFile = csv.writer(fp)
        pd.set_option('display.width', 1000)
        myFile.writerows(result)
        fp.close()
        df = pd.read_csv('report1.csv')
        df.to_csv("report1.csv",header=['USER_ID','HEART_RATE','CIGS PER DAY','SYSBP','DIABP','CHOLOSTROL','BMI','RISK FACTOR','DATE'],index=False)
        with open("report1.csv","rb") as f:
            file_data=f.read()
            print("file data in binary",file_data)
            file_name=f.name
            print("filename",file_name)
            msg.add_attachment(file_data,maintype="application",subtype="xlsx",filename=file_name)
        msg['To']=dm
        server.send_message(msg)
        del msg['To']
        del msg['Subject']
        msg.clear()
        return render_template('positive.html')
    
@app.route('/details',methods=['POST'])
def dets():
    global sex
    global age
    global currentSmoker
    global cigsPerDay
    global BPMeds
    global prevalentHyp
    global prevalentStroke
    global diabetes
    global totChol
    global sysBP
    global diaBP
    global BMI
    global glucose
    global m
    global id
    if request.method=='POST':
        sex=int(request.form['sex'])
        age=int(request.form['age'])
        currentSmoker = int(request.form['currentSmoker'])
        cigsPerDay = int(request.form['cigsPerDay'])
        BPMeds = int(request.form['BPMeds'])
        prevalentStroke = int(request.form['prevalentStroke'])
        prevalentHyp = int(request.form['prevalentHyp'])
        diabetes = int(request.form['diabetes'])
        totChol = int(request.form['totChol'])
        sysBP = int(request.form['sysBP'])
        diaBP = int(request.form['diaBP'])
        BMI = float(request.form['BMI'])
        glucose = int(request.form['glucose'])
        pm=request.form['pmail']
        dm=request.form['dmail']
        cur=mysql.connection.cursor()
        cur.execute("INSERT INTO user_details VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(sex,age,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,glucose,pm,dm,id))
        cur.connection.commit()
        return render_template("index.html")
    
if __name__=="__main__":
    app.run(debug=True,port=7384)