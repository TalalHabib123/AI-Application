import requests

def test1():
    url = 'http://localhost:5000/'
    response = requests.post(url)
    print(response.json())

def test():
    url = 'http://localhost:5000/predict'
    files = {'file': open('test_label_1.jpg', 'rb')}
    response = requests.post(url, files=files)
    print(response.json())
    
if __name__ == '__main__':
    test()