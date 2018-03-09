import lstm2
import time
import matplotlib.pyplot as plt

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data)#, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data)#, label='Prediction')
        plt.legend()
    plt.show()

def show_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])    
    plt.title('model train vs validation loss')    
    plt.ylabel('loss')    
    plt.xlabel('epoch')    
    plt.legend(['train', 'validation'], loc='upper right')    
    plt.show()

#Main Run Thread
if __name__=='__main__':
    global_start_time = time.time()
    epochs  = 500
    seq_len = 50
    out_dim = 10
    pre_len = 20
    
    print('> Loading data... ')
    
    X_train, y_train, X_test, y_test = lstm2.load_data('sp500.csv', seq_len, out_dim, False)
    
    print('> Data Loaded. Compiling...')
    
    model = lstm2.build_model([1, seq_len, 100, out_dim])
    
    history = model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05)
    
    show_history(history)
    
    predictions = lstm2.predict_sequences_multiple(model, X_test, seq_len, pre_len)
    #predicted = lstm.predict_sequence_full(model, X_test, seq_len)
    #predicted = lstm.predict_point_by_point(model, X_test)        
    
    print('Training duration (s) : ', time.time() - global_start_time)
    plot_results_multiple(predictions, y_test[:, 0], pre_len)