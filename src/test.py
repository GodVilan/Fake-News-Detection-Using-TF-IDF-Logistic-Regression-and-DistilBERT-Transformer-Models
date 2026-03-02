import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict_transformer import load_transformer, predict_text
m, t, d = load_transformer()
print(predict_text("NASA successfully launches Artemis mission", m, t, d))
print(predict_text("Aliens invade earth tomorrow!!!", m, t, d))
