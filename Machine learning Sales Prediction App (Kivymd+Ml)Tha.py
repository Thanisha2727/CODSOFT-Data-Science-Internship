from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.core.window import Window
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.filemanager import MDFileManager
from kivy.uix.scrollview import ScrollView
from kivy.metrics import dp
from kivy.utils import get_color_from_hex
from kivy.uix.image import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

Window.size = (360, 680)

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.data = None


        self.layout = MDBoxLayout(
            orientation='vertical',
            spacing=15,
            padding=[20, 30, 20, 50],
            md_bg_color=get_color_from_hex("#121212")  # Dark background
        )


        self.title = MDLabel(
            text="[b]Sales Predictor[/b]",
            halign="center",
            theme_text_color="Custom",
            text_color=get_color_from_hex("#FFFFFF"),
            font_style="H4",
            size_hint_y=None,
            height=dp(60),
            markup=True
        )


        self.file_card = MDCard(
            orientation='vertical',
            padding=15,
            size_hint=(0.92, None),
            height=dp(150),
            md_bg_color=get_color_from_hex("#1E1E1E"),  # Dark card background
            radius=[20],
            elevation=5,
            pos_hint={'center_x': 0.5}
        )
        self.file_label = MDLabel(
            text="Select a CSV file",
            halign="center",
            size_hint_y=None,
            height=dp(40),
            theme_text_color="Custom",
            text_color=get_color_from_hex("#BBDEFB")
        )
        self.load_btn = MDRaisedButton(
            text="Choose File",
            pos_hint={'center_x': 0.5},
            size_hint=(0.75, None),
            size=(dp(180), dp(50)),
            md_bg_color=get_color_from_hex("#0288D1"),
            text_color=(1, 1, 1, 1),
            font_size=16
        )
        self.load_btn.bind(on_release=self.show_file_manager)
        self.file_card.add_widget(self.file_label)
        self.file_card.add_widget(self.load_btn)


        self.preview_card = MDCard(
            orientation='vertical',
            padding=15,
            size_hint=(0.92, None),
            height=dp(220),
            md_bg_color=get_color_from_hex("#1E1E1E"),
            radius=[20],
            elevation=5,
            pos_hint={'center_x': 0.5}
        )
        scroll = ScrollView(size_hint=(1, None), size=(Window.width - 60, dp(140)))
        self.preview_label = MDLabel(
            text="Load a dataset to see preview",
            halign="left",
            size_hint_y=None,
            markup=True,
            theme_text_color="Custom",
            text_color=get_color_from_hex("#E0E0E0")
        )
        self.preview_label.bind(texture_size=self.preview_label.setter('size'))
        scroll.add_widget(self.preview_label)
        self.class_count_label = MDLabel(
            text="",
            halign="center",
            size_hint_y=None,
            height=dp(40),
            theme_text_color="Custom",
            text_color=get_color_from_hex("#B0BEC5")
        )
        self.preview_card.add_widget(scroll)
        self.preview_card.add_widget(self.class_count_label)


        self.button_layout = MDBoxLayout(
            orientation='horizontal',
            spacing=15,
            size_hint=(0.92, None),
            height=dp(70),
            pos_hint={'center_x': 0.5}
        )
        self.train_btn = MDRaisedButton(
            text="Train Model",
            size_hint=(0.5, None),
            size=(dp(150), dp(50)),
            md_bg_color=get_color_from_hex("#388E3C"),
            text_color=(1, 1, 1, 1),
            font_size=16,
            on_release=self.train_model,
            disabled=True
        )
        self.next_btn = MDRaisedButton(
            text="View Plot",
            size_hint=(0.5, None),
            size=(dp(150), dp(50)),
            md_bg_color=get_color_from_hex("#0288D1"),
            text_color=(1, 1, 1, 1),
            font_size=16,
            on_release=self.show_plot_and_next,
            disabled=True
        )
        self.button_layout.add_widget(self.train_btn)
        self.button_layout.add_widget(self.next_btn)


        self.result_card = MDCard(
            orientation='vertical',
            padding=15,
            size_hint=(0.92, None),
            height=dp(200),
            md_bg_color=get_color_from_hex("#1E1E1E"),
            radius=[20],
            elevation=5,
            pos_hint={'center_x': 0.5}
        )
        self.result_label = MDLabel(
            text="Model results will appear here",
            halign="left",
            size_hint_y=None,
            height=dp(180),
            theme_text_color="Custom",
            text_color=get_color_from_hex("#E0E0E0")
        )
        self.result_card.add_widget(self.result_label)


        self.layout.add_widget(self.title)
        self.layout.add_widget(self.file_card)
        self.layout.add_widget(self.preview_card)
        self.layout.add_widget(self.button_layout)
        self.layout.add_widget(self.result_card)
        self.add_widget(self.layout)


        self.file_manager = MDFileManager(
            select_path=self.load_csv,
            exit_manager=self.exit_file_manager,
            preview=False,
            ext=[".csv"]
        )

    def show_file_manager(self, *args):
        start_path = os.path.expanduser("~")
        self.file_manager.show(start_path)

    def exit_file_manager(self, *args):
        self.file_manager.close()

    def load_csv(self, path):
        try:
            self.data = pd.read_csv(path)
            self.file_label.text = f"Loaded: {os.path.basename(path)}"
            self.preview_label.text = str(self.data.head())
            self.class_count_label.text = f"Rows: {len(self.data)} | Columns: {len(self.data.columns)}"
            self.train_btn.disabled = False
            self.next_btn.disabled = False
        except Exception as e:
            self.preview_label.text = f"Error: {e}"
        self.exit_file_manager()

    def train_model(self, *args):
        if self.data is not None:
            try:
                X = self.data[['TV', 'Radio', 'Newspaper']]
                y = self.data['Sales']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                self.result_label.text = f"[b]R² Score:[/b] {r2:.2f}\n[b]MSE:[/b] {mse:.2f}"
            except Exception as e:
                self.result_label.text = f"Training error: {e}"

    def show_plot_and_next(self, instance):
        if self.data is not None:
            try:
                X = self.data[['TV', 'Radio', 'Newspaper']]
                y = self.data['Sales']
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X_test)

                plt.figure(figsize=(6, 4), facecolor='#121212')
                plt.scatter(y_test, y_pred, color='#BBDEFB', alpha=0.7)
                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='#F06292', linestyle='--')
                plt.xlabel('Actual Sales', color='#E0E0E0')
                plt.ylabel('Predicted Sales', color='#E0E0E0')
                plt.title('Prediction Accuracy', color='#FFFFFF')
                plt.grid(True, linestyle='--', alpha=0.5, color='#B0BEC5')
                plt.tight_layout()
                plt.gca().set_facecolor('#1E1E1E')
                plt.gca().spines['top'].set_color('#B0BEC5')
                plt.gca().spines['right'].set_color('#B0BEC5')
                plt.gca().spines['left'].set_color('#B0BEC5')
                plt.gca().spines['bottom'].set_color('#B0BEC5')
                plt.tick_params(colors='#E0E0E0')
                plot_path = "prediction_plot.png"
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()

                self.manager.get_screen('next').set_image(plot_path)
                self.manager.current = "next"

            except Exception as e:
                self.result_label.text = f"Plot error: {e}"

class NextScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = MDBoxLayout(
            orientation='vertical',
            padding=[20, 50, 20, 30],
            spacing=20,
            md_bg_color=get_color_from_hex("#121212")
        )
        label = MDLabel(
            text="[b]Prediction Results[/b]",
            halign="center",
            theme_text_color="Custom",
            text_color=get_color_from_hex("#FFFFFF"),
            font_style="H5",
            markup=True
        )
        self.image = Image(
            source="",
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(0.92, None),
            height=dp(320),
            pos_hint={'center_x': 0.5}
        )
        self.back_btn = MDRaisedButton(
            text="Back to Main",
            pos_hint={'center_x': 0.5},
            size_hint=(0.65, None),
            size=(dp(180), dp(50)),
            md_bg_color=get_color_from_hex("#F06292"),
            text_color=(1, 1, 1, 1),
            font_size=16,
            on_release=self.go_back
        )

        layout.add_widget(label)
        layout.add_widget(self.image)
        layout.add_widget(self.back_btn)
        self.add_widget(layout)

    def set_image(self, img_path):
        self.image.source = img_path
        self.image.reload()

    def go_back(self, *args):
        self.manager.current = 'main'

class SalesPredictionApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Pink"
        self.theme_cls.theme_style = "Dark"
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(NextScreen(name='next'))
        return sm

if __name__ == '__main__':
    SalesPredictionApp().run()