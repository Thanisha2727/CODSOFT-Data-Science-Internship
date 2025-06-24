from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivy.uix.scrollview import ScrollView
from kivy.metrics import dp
from kivy.uix.image import AsyncImage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import os
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton

class IrisClassifierScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.dialog = None
        self.train_model()
        self.build_ui()

    def train_model(self):
        try:

            try:

                df = pd.read_csv('iris.csv')
            except FileNotFoundError:

                from sklearn.datasets import load_iris
                iris = load_iris()
                df = pd.DataFrame(
                    data=iris.data,
                    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
                )
                df['species'] = iris.target_names[iris.target]


            X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
            y = df['species']


            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)


            X_train, _, y_train, _ = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)


            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)
        except Exception as e:
            self.show_error_dialog(f"Error loading or training model: {str(e)}")

    def build_ui(self):
        layout = MDBoxLayout(
            orientation='vertical',
            padding=dp(20),
            spacing=dp(15),
            adaptive_height=True
        )

        layout.add_widget(MDLabel(
            text="🌸 Iris Flower Classifier 🌸",
            halign="center",
            theme_text_color="Primary",
            font_style="H4",
            size_hint=(1, None),
            height=dp(60)
        ))


        layout.add_widget(AsyncImage(
            source='https://raw.githubusercontent.com/python/cpython/main/Doc/readme-images/iris-machinelearning.png',
            size_hint=(1, 0.3),
            pos_hint={'center_x': 0.5},
            allow_stretch=True,
            keep_ratio=True
        ))


        input_fields = [
            ("sepal_length", "Sepal Length (cm)"),
            ("sepal_width", "Sepal Width (cm)"),
            ("petal_length", "Petal Length (cm)"),
            ("petal_width", "Petal Width (cm)")
        ]

        for field_name, hint_text in input_fields:
            text_field = MDTextField(
                hint_text=hint_text,
                input_filter="float",
                size_hint=(0.9, None),
                height=dp(48),
                pos_hint={'center_x': 0.5},
                mode="rectangle",
                required=True,
                helper_text=f"Enter valid {hint_text.lower()} (0.1-10.0)",
                helper_text_mode="on_error"
            )
            setattr(self, field_name, text_field)
            layout.add_widget(text_field)


        button_layout = MDBoxLayout(
            padding=dp(10),
            spacing=dp(10),
            size_hint=(1, None),
            height=dp(60)
        )


        predict_button = MDRaisedButton(
            text="Predict Species",
            size_hint=(0.5, None),
            height=dp(50),
            pos_hint={'center_x': 0.5},
            on_press=self.predict_species
        )

        clear_button = MDRaisedButton(
            text="Clear",
            size_hint=(0.5, None),
            height=dp(50),
            pos_hint={'center_x': 0.5},
            on_press=self.clear_inputs
        )

        button_layout.add_widget(predict_button)
        button_layout.add_widget(clear_button)


        self.result_label = MDLabel(
            text="Prediction: None",
            halign="center",
            theme_text_color="Primary",
            font_style="H6",
            size_hint=(1, None),
            height=dp(50)
        )


        for widget in [button_layout, self.result_label]:
            layout.add_widget(widget)


        scroll = ScrollView(size_hint=(1, 1))
        scroll.add_widget(layout)
        self.add_widget(scroll)

    def validate_input(self, value):
        try:
            val = float(value)
            return 0.1 <= val <= 10.0
        except ValueError:
            return False

    def predict_species(self, instance):
        try:

            inputs = []
            for field in [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]:
                if not field.text or not self.validate_input(field.text):
                    field.error = True
                    raise ValueError(f"Invalid input for {field.hint_text}")
                field.error = False
                inputs.append(float(field.text))

            inputs_scaled = self.scaler.transform([inputs])
            prediction = self.model.predict(inputs_scaled)
            species = self.label_encoder.inverse_transform(prediction)[0]
            self.result_label.text = f"Prediction: {species.capitalize()}"
        except Exception as e:
            self.show_error_dialog(str(e))

    def clear_inputs(self, instance):
        for field in [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]:
            field.text = ""
            field.error = False
        self.result_label.text = "Prediction: None"

    def show_error_dialog(self, error_message):
        if not self.dialog:
            self.dialog = MDDialog(
                title="Error",
                text=error_message,
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_press=lambda x: self.dialog.dismiss()
                    )
                ]
            )
        self.dialog.text = error_message
        self.dialog.open()

class IrisClassifierApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Purple"
        self.theme_cls.theme_style = "Light"
        self.theme_cls.material_style = "M3"
        return IrisClassifierScreen()

if __name__ == '__main__':
    IrisClassifierApp().run()
