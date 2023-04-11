#!/usr/bin/env python
# coding: utf-8

# # 1. Import Library

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


loan_data = pd.read_csv('Data/loan_data_2007_2014.csv', index_col = 0)


# In[3]:


loan_data.head()


# In[4]:


loan_data.shape


# In[5]:


loan_data.info()


# ------------

# # 2.Target Variable

# Dikarenakan project ini untuk mengetahui bad loan & good loan, maka perlu dibuat feature baru, yaitu target variable yang merepresentasikan bad loan (sebagai 1) dan good loan (sebagai 0).

# In[6]:


loan_data['loan_status'].unique()


# In[7]:


# Membuat feature baru yaitu good_bad sebagai target variable,
# Jika loan_statusnya 'Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)' 
# akan dianggap sebagai bad_loan atau 1 dan nilai selain itu akan dianggap good loan atau 0

loan_data['good_bad'] = np.where(loan_data.loc[:,'loan_status'].isin(['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off'])
                        , 1 , 0)


# In[8]:


# Melihat distribusi 0 dan 1
loan_data['good_bad'].value_counts()


# In[9]:


# 1=Bad 0=Good
loan_data[['loan_status', 'good_bad']]


# In[10]:


loan_data['good_bad'].value_counts(normalize=True)


# In[11]:


# Melihat feature apa saja yang memiliki missing value lebih dari 50%
missing_values = pd.DataFrame(loan_data.isnull().sum()/loan_data.shape[0])
missing_values = missing_values[missing_values.iloc[:,0] > 0.50]
missing_values.sort_values([0], ascending=False)


# In[12]:


# Drop feature tersebut
loan_data.dropna(thresh = loan_data.shape[0]*0.5, axis=1, inplace=True)


# In[13]:


# Cek missing values yang tersisa
missing_values = pd.DataFrame(loan_data.isnull().sum()/loan_data.shape[0])
missing_values = missing_values[missing_values.iloc[:,0]>0.50]
missing_values.sort_values((0), ascending=True)


# ---------

# # 3.Data Splitting

# In[14]:


loan_data.shape


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


# Membagi data menjadi 80/20 dengan menyamakan distribusi dari bad loans di test set dengan train set.
X = loan_data.drop('good_bad', axis=1)
y = loan_data['good_bad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify= y, random_state=42)


# In[18]:


X_train.shape, X_test.shape


# In[19]:


y_train.value_counts(normalize=True)


# In[20]:


# Distribusi y_test sudah sama persis dengan y_train
y_test.value_counts(normalize=True)


# --------------------

# # 4.Data Cleaning

# In[21]:


# Terdapat 53 kolom, bagaimana untuk mengetahui kolom apa saja yang memiliki data kotor?
X_train.shape


# In[22]:


# Dapat dilakukan print untuk semua unique values kolom, sehingga dapat di cek satu-satu
# unique values apa saja yang kotor.

for col in X_train.select_dtypes(include= ['object','bool']).columns:
    print(col)
    print(X_train[col].unique())
    print()


# In[23]:


# Kolom/feature yang harus di cleaning
col_need_to_clean = ['term', 'emp_length', 'issue_d', 'earliest_cr_line', 'last_pymnt_d', 
                    'next_pymnt_d', 'last_credit_pull_d']


# In[24]:


# Convert data type menjadi numeric 
X_train['term'] = pd.to_numeric(X_train['term'].str.replace(' months', ''))


# In[25]:


X_train['term']


# In[26]:


# Cek values apa saja yang harus di cleaning
X_train['emp_length'].unique()


# In[27]:


X_train['emp_length'] = X_train['emp_length'].str.replace('\+ years', '')
X_train['emp_length'] = X_train['emp_length'].str.replace(' years', '')
X_train['emp_length'] = X_train['emp_length'].str.replace('< 1 year', str(0))
X_train['emp_length'] = X_train['emp_length'].str.replace(' year', '')

X_train['emp_length'].fillna(value = 0, inplace=True)
X_train['emp_length'] = pd.to_numeric(X_train['emp_length'])


# In[28]:


X_train['emp_length']


# In[29]:


# Cek feature date
col_date = ['issue_d', 'earliest_cr_line', 'last_pymnt_d',
                    'next_pymnt_d', 'last_credit_pull_d']

X_train[col_date]


# In[30]:


# Mengganti yang bertipe data object ke datetimes64
X_train['issue_d'] = pd.to_datetime(X_train['issue_d'], format = ("%b-%y"))


# In[31]:


# Mengganti yang bertipe data object ke datetimes64
X_train['earliest_cr_line'] = pd.to_datetime(X_train['earliest_cr_line'], format = ("%b-%y"))
X_train['last_pymnt_d'] = pd.to_datetime(X_train['last_pymnt_d'], format = ("%b-%y"))
X_train['next_pymnt_d'] = pd.to_datetime(X_train['next_pymnt_d'], format = ("%b-%y"))
X_train['last_credit_pull_d'] = pd.to_datetime(X_train['last_credit_pull_d'], format = ("%b-%y"))


# In[32]:


# Lakukan hal yang sama untuk X_test
X_test['term'] = pd.to_numeric(X_test['term'].str.replace(' months', ''))

X_test['emp_length'] = X_test['emp_length'].str.replace('\+ years', '')
X_test['emp_length'] = X_test['emp_length'].str.replace(' years', '')
X_test['emp_length'] = X_test['emp_length'].str.replace('< 1 year', str(0))
X_test['emp_length'] = X_test['emp_length'].str.replace(' year', '')

X_test['emp_length'].fillna(value = 0, inplace=True)
X_test['emp_length'] = pd.to_numeric(X_test['emp_length'])


# In[33]:


# Mengganti yang bertipe data object ke datetimes64
X_test['issue_d'] = pd.to_datetime(X_test['issue_d'], format = ("%b-%y"))


# In[34]:


# Mengganti yang bertipe data object ke datetimes64
X_test['earliest_cr_line'] = pd.to_datetime(X_test['earliest_cr_line'], format = ("%b-%y"))
X_test['last_pymnt_d'] = pd.to_datetime(X_test['last_pymnt_d'], format = ("%b-%y"))
X_test['next_pymnt_d'] = pd.to_datetime(X_test['next_pymnt_d'], format = ("%b-%y"))
X_test['last_credit_pull_d'] = pd.to_datetime(X_test['last_credit_pull_d'], format = ("%b-%y"))


# In[35]:


X_train[col_date]


# In[36]:


# Check apakah berhasil di cleaning
X_test[col_need_to_clean].info()


# -----------------

# # 5. Feature Engineering

# In[37]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[38]:


# Kolom yang akan di gunakan
col_need_to_clean


# In[39]:


X_train = X_train[col_need_to_clean]
X_test = X_test[col_need_to_clean]


# In[40]:


# tidak dibutuhkan untuk modelling
del X_train['next_pymnt_d']
del X_test['next_pymnt_d']


# In[41]:


X_train.shape, X_test.shape


# In[42]:


from datetime import date

date.today().strftime('%Y-%m-%d')


# In[43]:


# feature engineering untuk date columns
def date_columns(df, column):
    today_date = pd.to_datetime(date.today().strftime('%Y-%m-%d'))
    df[column] = pd.to_datetime(df[column], format = "%b-%y")
    df['mths_since_' + column] = round(pd.to_numeric((today_date - df[column]) / np.timedelta64(1, 'M')))
    df.drop(columns = [column], inplace=True)
    
# apply to X_train
date_columns(X_train, 'earliest_cr_line')
date_columns(X_train, 'issue_d')
date_columns(X_train, 'last_pymnt_d')
date_columns(X_train, 'last_credit_pull_d')


# In[44]:


# apply to X_test
date_columns(X_test, 'earliest_cr_line')
date_columns(X_test, 'issue_d')
date_columns(X_test, 'last_pymnt_d')
date_columns(X_test, 'last_credit_pull_d')


# In[45]:


X_test.isnull().sum()


# In[46]:


X_train.isnull().sum()


# In[47]:


X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_test.median(), inplace=True)


# In[48]:


X_test.isnull().sum()


# In[49]:


X_train.isnull().sum()


# ------------

# # 6. Modelling

# In[74]:


from sklearn.neighbors import KNeighborsClassifier as KNN #pembuatan model KNeighbors Classifier


# In[79]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score #evaluasi dan performa


# In[77]:


KNN_model = KNN()
KNN_model = KNN_model.fit(X_train, y_train)


# In[80]:


#melakukan prediksi pada data test
y_pred_KNN = KNN_model.predict(X_test)

#check performa dari model menggunakan classification_report
print(classification_report(y_test, y_pred_KNN))

#evaluasi model
acc_score_KNN = round(accuracy_score(y_pred_KNN, y_test), 3)
print('Accuracy model KNeighbors Classifier : ', acc_score_KNN)


# In[86]:


#confussion Matrix
cm_KNN = confusion_matrix(y_pred_KNN, y_test)
sns.heatmap(cm_KNN, annot = True, fmt='d', cmap=plt.cm.Blues);

plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.show()


# In[ ]:




