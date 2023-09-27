import matplotlib.pyplot as plt

def plot_history(history):
   fig, axs = plt.subplots (2,1,figsize=(10,6))
   axs[0].plot (history.history['loss'],label='Training Loss')
   axs[0].plot (history.history['val_loss'],label='Validation Loss')
   axs[1].plot (history.history['accuracy'],label='Training Accurcay')
   axs[1].plot (history.history['val_accuracy'],label='Validation Accurcay')
   axs[0].legend()
   axs[1].legend()
   plt.show()


