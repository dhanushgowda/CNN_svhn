from flask import Flask, make_response, render_template
from eval import get_predictions
from functools import update_wrapper

app = Flask(__name__)


def nocache(f):
    def new_func(*args, **kwargs):
        resp = make_response(f(*args, **kwargs))
        resp.cache_control.no_cache = True
        return resp

    return update_wrapper(new_func, f)


@app.route("/")
@nocache
def run_eval():
    x, y = get_predictions()
    pred = [i[0] for i in x]
    actual = [i[1] for i in x]
    z = []
    for i in range(0, len(y)):
        z.append([pred[i], actual[i], y[i]])
    # print(z)
    # a = z[0:4]
    # b = z[5:9]
    # c = z[10:14]
    # d = z[15:19]
    a,b,c,d = (z[0:5], z[5:10], z[10:15], z[15:20])
    return render_template("op.html", data=z, new_data=zip(a, b, c, d))


if __name__ == "__main__":
    app.run(host='0.0.0.0')
