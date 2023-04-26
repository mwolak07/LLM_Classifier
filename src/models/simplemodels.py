import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn import svm
import gensim
from gensim.models import fasttext
from src.util import load_data, cd_to_executing_file
import re



from src.dataset import LLMClassifierDataset

print("CODE HAS STARTED")

#returns predictions on test data using LogReg model
def generateLogisticRegression(data, classes, predictMe):
    model = LogisticRegression(penalty='l2', max_iter = 250).fit(data, classes)
    return model.predict(predictMe)
    
#returns predictions on test data using Naive Bayes model
def generateNaiveBayes(data, classes, predictMe):
    model = GaussianNB().fit(data,classes)
    return model.predict(predictMe)
    
#prec, f1, recall, aucroc, returns all 4 in tuple
def runMetrics(predicted, actual):
    prec = sk.metrics.precision_score(actual, predicted)
    f1 = sk.metrics.f1_score(actual, predicted)
    recall = sk.metrics.recall_score(actual, predicted)
    aucroc = sk.metrics.roc_auc_score(actual, predicted)
    print("Precision score of the model:", prec)
    print("F1 score of the model", f1)
    print("Recall score of the model", recall)
    print("AUCROC of the model", aucroc)
    return (prec, f1, recall, aucroc)

#to test the iris data set, wont be used on real dataset
def runMetricsMulticlass(predicted, actual):
    prec = sk.metrics.precision_score(actual, predicted, average = 'micro')
    f1 = sk.metrics.f1_score(actual, predicted, average = 'micro')
    recall = sk.metrics.recall_score(actual, predicted, average = 'micro')
    print("Precision score of the model:", prec)
    print("F1 score of the model", f1)
    print("Recall score of the model", recall)
    return [prec, f1, recall]


print("making fastext")
model_path = "../../data/fasttext/wiki.en.bin"
gensimModel = gensim.models.fasttext.load_facebook_model(model_path)
# gensimModel = gensim.models.FastText.load(model_path)

print("made it")

#data should be [[[sent vect][sent vect]],[list of sent in response],[list of sent in response],response,response], 
# where n is number of responses, X1 is number of sentences in response, X2 is word vector
def runfasttext(data):
    fasttextout = []#will need to be an array :*(
    for element in data:
        words = element.split(' ')
        vectors = []
        for word in words:
            vectors.append(gensimModel.wv[word])
        fasttextout.append(vectors)
    return fasttextout
    
#data should be [[[sent vect][sent vect]],[list of sent in response],[list of sent in response],response,response], 
#where n is number of responses, X1 is number of sentences in response, X2 is word vector
#result of this should hopefully standardize the number of sent vect per response, 
#so the array is (n,max_seq_length,gensim.model.vector_size)
def padInput(data, max_seq_length):   
    print(str(max_seq_length) + "MAXXXXXXXX")
    padData = []
    vecSize = len(data[0][0])
    for i in range(len(data)):
        #amount of missing sentence vectors
        fixedCol = data[i]
        
        #we are padding j times, so that len(fixedColumn) = max_seq
        for j in range(max_seq_length - len(data[i])):
            fixedCol.append(np.zeros(vecSize))
        padData.append(fixedCol)
    return padData


print("database")
# dbdata = LLMClassifierDataset(db_path="src/models/test_short_prompts.sqlite3", load_to_memory=False)
cd_to_executing_file(__file__)
trainData, testData, trainLabels, testLabels = load_data("../../data/bloom_1_1B/dev_v2.1_short_prompts_train.sqlite3",
                                                         "../../data/bloom_1_1B/dev_v2.1_short_prompts_test.sqlite3")
maxV = max(max(len(seq) for seq in trainData), max(len(seq) for seq in testData))
trainData = runfasttext(trainData)
testData = runfasttext(testData)
trainData = padInput(trainData, maxV)
testData = padInput(testData, maxV)


# dbdata = dbdata.tolist()

print("dbdata is ready")

# allData = np.array([])
# allLabels = np.array([])
# for i in range(len(dbdata)):
#     allData = np.append(allData,dbdata[i][0])
#     allLabels = np.append(allLabels,dbdata[i][1])
    
#this should make a marco and llm dataset
#will have N elements, each element will be the prompt string with the answer strong attatched at the end

# allData = runfasttext(allData)
# allData = np.array(padInput(allData))
# print("train test split")
# p = np.random.permutation(len(allData))
# allData = allData[p]
# allLabels = allLabels[p]
# trainData = allData[0:int(3*len(allData)/4)]
# trainLabels = allLabels[0:int(3*len(allData)/4)]
# testData = allData[int(3*len(allData)/4):len(allData)]
# testLabels = allLabels[int(3*len(allData)/4):len(allData)]

print(np.shape(trainData))
print(np.shape(testData))

#flatten a dimension for models because apparently needed
flatTrain = []
flatTest = []
for i in range(len(trainData)):
    flatTrain.append(np.mean(trainData[i],axis = 0))
for i in range(len(testData)):
    flatTest.append(np.mean(testData[i],axis = 0))
flatTrain = np.array(flatTrain)
flatTest = np.array(flatTest)

print("models about to run")

predClassLog = generateLogisticRegression(flatTrain,trainLabels,flatTest)
print(predClassLog)
runMetrics(predClassLog,testLabels)
predClassBayes = generateNaiveBayes(flatTrain,trainLabels,flatTest)
print(predClassBayes)
runMetrics(predClassBayes,testLabels)




# logRegModel = LogisticRegression().fit(flatTrain, trainLabels)
# explainer = shap.LinearExplainer(logRegModel, flatTrain)
# shap_values = explainer.shap_values(flatTrain)
# bapValues = explainer(flatTest)
    
# shap.initjs()
# shap.plots.beeswarm(bapValues)



# guassModel = GaussianNB(penalty='l2', max_iter = 250).fit(flatTrain, trainLabels)
# explainerNB = shap.LinearExplainer(guassModel, flatTrain)
# shap_valuesNB = explainerNB.shap_values(flatTrain)
# bapValuesNB = explainerNB(flatTest)
    
# shap.initjs()
# shap.plots.beeswarm(bapValuesNB)

