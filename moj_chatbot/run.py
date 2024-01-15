from flask import Flask, request, abort
from flask_restx import Resource, Api, fields
from model import ChatbotModel
from waitress import serve


app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="MoJ Chatbot",
    description="API for training and querying a chatbot",
)

model = ChatbotModel()

chatbot_ns = api.namespace("chatbot", description="Used to train and query the chatbot")

question_model = api.model(
    "Question",
    {
        "SessionID": fields.String,
        "SystemID": fields.String,
        "QuestionText": fields.String,
    },
)
intent_question_model = api.model(
    "IntentQuestion",
    {
        "IntentID": fields.String,
        "QuestionID": fields.String,
        "QuestionText": fields.String,
    },
)
predicted_intent = api.model(
    "PredictedIntent",
    {
        "IntentID": fields.String,
        "Confidence": fields.Float,
    },
)
resource_fields = api.model(
    "Intent",
    {
        "SystemID": fields.String,
        "AddedItems": fields.List(fields.Nested(intent_question_model)),
        "EditedItems": fields.List(fields.Nested(intent_question_model)),
        "DeletedItems": fields.List(fields.Nested(intent_question_model)),
    },
)


@chatbot_ns.route("/train")
class Train(Resource):
    @api.doc(responses={200: "Success", 400: "Error"})
    @api.expect([resource_fields])
    def post(self):
        try:
            data = request.get_json()
            model.train(data)
            return "Success", 200
        except Exception as e:
            abort(400, str(e))


@chatbot_ns.route("/query")
class Query(Resource):
    @api.response(200, "Success", predicted_intent)
    @api.response(400, "Error")
    @api.expect(question_model)
    def post(self):
        try:
            data = request.get_json()
            session_id = data["SessionID"]
            text = data["QuestionText"]
            system_id = data["SystemID"]
            intent_id, similarity = model.query(system_id, text, 1)[0]
            return {
                "PredictedIntent": {
                    "IntentID": intent_id,
                    "SessionID": session_id,
                    "Confidence": similarity,
                }
            }
        except Exception as e:
            abort(400, str(e))


@chatbot_ns.route("/query/<int:top_k>")
class QueryTopK(Resource):
    @api.response(200, "Success", [predicted_intent])
    @api.response(400, "Error")
    @api.expect(question_model)
    def post(self, top_k):
        try:
            data = request.get_json()
            session_id = data["SessionID"]
            text = data["QuestionText"]
            system_id = data["SystemID"]
            intent_sim_scores = model.query(system_id, text, top_k)
            return [
                {
                    "PredictedIntent": {
                        "IntentID": intent_id,
                        "SessionID": session_id,
                        "Confidence": similarity,
                    }
                }
                for intent_id, similarity in intent_sim_scores
            ]
        except Exception as e:
            abort(400, str(e))


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=7000, debug=True)
    serve(app=app, host="0.0.0.0", port=7000)
