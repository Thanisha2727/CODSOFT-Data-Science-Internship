from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics.texture import Texture
from io import BytesIO

class MainScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = MDBoxLayout(
            orientation='vertical',
            padding=20,
            spacing=20,
            md_bg_color=[0.1, 0.1, 0.1, 1]
        )

        self.title_label = MDLabel(
            text="Credit Card Fraud Detection",
            halign="center",
            theme_text_color="Custom",
            text_color=[1, 1, 1, 1],
            font_style="H4",
            bold=True
        )

        self.load_button = MDRaisedButton(
            text="Load CSV File",
            pos_hint={'center_x': 0.5},
            on_release=self.show_file_manager,
            md_bg_color=[0.2, 0.4, 0.8, 1],
            text_color=[1, 1, 1, 1],
            size_hint=(0.8, None),
            height=50
        )

        self.train_button = MDRaisedButton(
            text="Train Model",
            pos_hint={'center_x': 0.5},
            on_release=self.train_model,
            disabled=True,
            md_bg_color=[0.2, 0.4, 0.8, 1],
            text_color=[1, 1, 1, 1],
            size_hint=(0.8, None),
            height=50
        )

        self.layout.add_widget(self.title_label)
        self.layout.add_widget(self.load_button)
        self.layout.add_widget(self.train_button)
        self.add_widget(self.layout)

        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            ext=['.csv']
        )

        self.df = None
        self.report_data = None

    def show_file_manager(self, *args):
        self.file_manager.show(os.path.expanduser("~"))

    def exit_manager(self, *args):
        self.file_manager.close()

    def select_path(self, path):
        self.exit_manager()
        try:
            self.df = pd.read_csv(path)
            self.train_button.disabled = False
        except Exception as e:
            print(f"Error loading file: {str(e)}")

    def train_model(self, *args):
        if self.df is None:
            return

        try:
            scaler = StandardScaler()
            self.df[['Time', 'Amount']] = scaler.fit_transform(self.df[['Time', 'Amount']])

            fraud = self.df[self.df['Class'] == 1]
            genuine = self.df[self.df['Class'] == 0].sample(n=len(fraud), random_state=42)
            balanced_df = pd.concat([fraud, genuine])
            balanced_df = balanced_df.sample(frac=1, random_state=42)

            X = balanced_df.drop('Class', axis=1)
            y = balanced_df['Class']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.report_data = report

            # Navigate to ResultScreen and pass report data
            self.manager.get_screen('result').update_report(report)
            self.manager.current = 'result'
        except Exception as e:
            print(f"Error training model: {str(e)}")

class ResultScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = MDBoxLayout(
            orientation='vertical',
            padding=20,
            spacing=20,
            md_bg_color=[0.1, 0.1, 0.1, 1]
        )

        self.result_label = MDLabel(
            text="Classification Report will appear here",
            halign="center",
            theme_text_color="Custom",
            text_color=[0.9, 0.9, 0.9, 1],
            size_hint=(1, 0.6)
        )

        self.next_button = MDRaisedButton(
            text="Show Plot",
            pos_hint={'center_x': 0.5},
            on_release=self.show_plot,
            md_bg_color=[0.6, 0.2, 0.8, 1],
            text_color=[1, 1, 1, 1],
            size_hint=(0.8, None),
            height=50
        )

        self.back_button = MDRaisedButton(
            text="Back",
            pos_hint={'center_x': 0.5},
            on_release=self.go_back,
            md_bg_color=[0.8, 0.2, 0.2, 1],
            text_color=[1, 1, 1, 1],
            size_hint=(0.8, None),
            height=50
        )

        self.layout.add_widget(self.result_label)
        self.layout.add_widget(self.next_button)
        self.layout.add_widget(self.back_button)
        self.add_widget(self.layout)

        self.report_data = None

    def update_report(self, report):
        self.report_data = report
        self.result_label.text = self.format_report(report)

    def format_report(self, report):
        return (f"Class 0 (Genuine):\n"
                f"  Precision: {report['0']['precision']:.2f}\n"
                f"  Recall: {report['0']['recall']:.2f}\n"
                f"  F1-Score: {report['0']['f1-score']:.2f}\n"
                f"Class 1 (Fraud):\n"
                f"  Precision: {report['1']['precision']:.2f}\n"
                f"  Recall: {report['1']['recall']:.2f}\n"
                f"  F1-Score: {report['1']['f1-score']:.2f}\n"
                f"Accuracy: {report['accuracy']:.2f}")

    def show_plot(self, *args):
        if self.report_data is None:
            self.result_label.text = "No report data available."
            return

        # Create a new screen for the plot
        plot_screen = self.manager.get_screen('plot')
        plot_screen.update_plot(self.report_data)
        self.manager.current = 'plot'

    def go_back(self, *args):
        self.manager.current = 'main'

class PlotScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = MDBoxLayout(
            orientation='vertical',
            padding=20,
            spacing=20,
            md_bg_color=[0.1, 0.1, 0.1, 1]
        )

        self.plot_image = Image(size_hint=(1, 0.6))
        self.back_button = MDRaisedButton(
            text="Back",
            pos_hint={'center_x': 0.5},
            on_release=self.go_back,
            md_bg_color=[0.8, 0.2, 0.2, 1],
            text_color=[1, 1, 1, 1],
            size_hint=(0.8, None),
            height=50
        )

        self.layout.add_widget(self.plot_image)
        self.layout.add_widget(self.back_button)
        self.add_widget(self.layout)

    def update_plot(self, report_data):
        # Create Matplotlib plot
        fig, ax = plt.subplots()
        metrics = ['precision', 'recall', 'f1-score']
        genuine = [report_data['0'][m] for m in metrics]
        fraud = [report_data['1'][m] for m in metrics]

        x = range(len(metrics))
        width = 0.35
        ax.bar([i - width/2 for i in x], genuine, width, label='Genuine (Class 0)', color='#36A2EB')
        ax.bar([i + width/2 for i in x], fraud, width, label='Fraud (Class 1)', color='#FF6384')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Scores')
        ax.set_title('Classification Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()

        # Convert plot to Kivy texture
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = plt.imread(buf)
        texture = Texture.create(size=(img.shape[1], img.shape[0]))
        texture.blit_buffer(img.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        self.plot_image.texture = texture
        plt.close(fig)

    def go_back(self, *args):
        self.manager.current = 'result'

class FraudDetectionApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.theme_style = "Dark"
        Window.size = (400, 600)

        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(ResultScreen(name='result'))
        sm.add_widget(PlotScreen(name='plot'))
        return sm

if __name__ == '__main__':
    FraudDetectionApp().run()