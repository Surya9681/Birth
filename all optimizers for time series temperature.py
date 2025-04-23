import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_temperature_data(n=100, seed=42):
    np.random.seed(seed)
    time = pd.date_range(start='2024-01-01', periods=n, freq='H')  
    temperature = 20 + np.cumsum(np.random.randn(n)) * 0.5  
    return pd.DataFrame({'Time': time, 'Temperature': temperature})

def sgd_momentum(df, alpha=0.1, beta=0.9):
    velocity = 0
    prev_value = df['Temperature'].iloc[0]
    values = [prev_value]

    for temp in df['Temperature'][1:]:
        gradient = prev_value - temp
        velocity = beta * velocity + alpha * gradient  # Momentum update
        new_value = prev_value - velocity
        values.append(new_value)
        prev_value = new_value

    return values

def adagrad(df, alpha=0.1, epsilon=1e-8):
    grad_squared = 0
    prev_value = df['Temperature'].iloc[0]
    values = [prev_value]

    for temp in df['Temperature'][1:]:
        gradient = prev_value - temp
        grad_squared += gradient ** 2
        adjusted_alpha = alpha / (np.sqrt(grad_squared) + epsilon)
        new_value = prev_value - adjusted_alpha * gradient
        values.append(new_value)
        prev_value = new_value

    return values

def rmsprop(df, alpha=0.1, beta=0.9, epsilon=1e-8):
    grad_squared_avg = 0
    prev_value = df['Temperature'].iloc[0]
    values = [prev_value]

    for temp in df['Temperature'][1:]:
        gradient = prev_value - temp
        grad_squared_avg = beta * grad_squared_avg + (1 - beta) * (gradient ** 2)
        adjusted_alpha = alpha / (np.sqrt(grad_squared_avg) + epsilon)
        new_value = prev_value - adjusted_alpha * gradient
        values.append(new_value)
        prev_value = new_value

    return values

def adam(df, alpha=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m, v = 0, 0
    t = 0
    prev_value = df['Temperature'].iloc[0]
    values = [prev_value]

    for temp in df['Temperature'][1:]:
        t += 1
        gradient = prev_value - temp

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        new_value = prev_value - (alpha * m_hat) / (np.sqrt(v_hat) + epsilon)
        values.append(new_value)
        prev_value = new_value

    return values

def plot_optimizers(df, alpha=0.1):
    plt.figure(figsize=(12, 6))

    plt.scatter(df['Time'], df['Temperature'], label='Original Temperature', color='black', alpha=0.5)

    df['SGD_Momentum'] = sgd_momentum(df, alpha)
    df['Adagrad'] = adagrad(df, alpha)
    df['RMSprop'] = rmsprop(df, alpha)
    df['Adam'] = adam(df, alpha)

    plt.plot(df['Time'], df['SGD_Momentum'], label='SGD with Momentum', linestyle='dotted', color='blue')
    plt.plot(df['Time'], df['Adagrad'], label='Adagrad', linestyle='dashed', color='green')
    plt.plot(df['Time'], df['RMSprop'], label='RMSprop', linestyle='dashdot', color='purple')
    plt.plot(df['Time'], df['Adam'], label='Adam', linestyle='solid', color='red')

    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.title('Comparison of Optimization Algorithms for Temperature Smoothing')
    plt.grid()
    plt.show()

# Run the pipeline
df = generate_temperature_data()
plot_optimizers(df, alpha=0.05)
