import flask
import pandas as pd

import cmu_etim.cake_or_not

if __name__ == '__main__':
    regressor = cmu_etim.cake_or_not.MediumCake()
    tr, trl, tst, tstl = regressor.recipes()
    regressor.set_test_data(tst, tstl['title'])
    regressor.train(pd.concat([tr, tst]), pd.concat(
        [trl['#cakeweek'], tstl['#cakeweek']]))

    # Create the API
    app = flask.Flask('API')

    @app.route('/', methods=['GET'])
    def etim():
        return flask.render_template('etim.html')

    @app.route('/query')
    def query():
        word = flask.request.args.get('word').lower()
        is_cake = regressor.predict_title(word)
        return flask.jsonify({'is_cake': bool(is_cake)})

    app.run(port=8080)
