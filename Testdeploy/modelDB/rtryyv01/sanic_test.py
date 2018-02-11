
from sanic import Sanic

app = Sanic()

@app.route("/sanicTest")
async def test(request):
    print('dfvcsdfvdfv')
    return "csdcdfvd"




if __name__ =="__main__":
    app.run(host="0.0.0.0", port=7071)