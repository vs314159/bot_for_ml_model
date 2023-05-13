from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from ml_model.predictor import create_sample, predict, MODEL_MAPE
from rasa_sdk.events import AllSlotsReset



class ActionPredict(Action):
    def name(self) -> Text:
        return "action_predict"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dict_user_data = {
            'district2': tracker.get_slot('district2'),
            'numberOfRooms': tracker.get_slot('numberOfRooms'),
            'appartment_metrage': tracker.get_slot('appartment_metrage'),
            'kitchen_metrage': tracker.get_slot('kitchen_metrage'),
            'bathroom_metrage': tracker.get_slot('bathroom_metrage'),
            'floor': tracker.get_slot('floor'),
            'build_type': tracker.get_slot('build_type'),
            'wall_type': tracker.get_slot('wall_type'),
            'is_app_complex': tracker.get_slot('is_app_complex'),
            'is_boiler': tracker.get_slot('is_boiler'),
            'is_air_cond': tracker.get_slot('is_air_cond'),
            'infrastructure': tracker.get_slot('infrastructure'),
            'is_subway': tracker.get_slot('is_subway'),
            'priceCurrency_offers': tracker.get_slot('priceCurrency_offers'),
            'is_ukr_lang': tracker.get_slot('is_ukr_lang'),
            'is_agency': tracker.get_slot('is_agency'),
        }

        sample = create_sample(dict_user_data)
        prediction = (10 ** predict(sample)[0])//1
        dispatcher.utter_message(text=f"Отже, вартість оренди буде приблизно {prediction} грн")
        return []


class ResetAllSlots(Action):
    def name(self) -> Text:
        return "action_reset_all_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [AllSlotsReset()]
