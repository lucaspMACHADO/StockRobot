from matplotlib import pyplot as plt
import os


def plot(b_money, s_money, stock_name, stock_price):
    fig = plt.figure()

    plt.title('VOL' + stock_name)
    plt.xlabel('Time')
    plt.ylabel('Gain')

    plt.plot(range(len(b_money)), b_money, linestyle='solid', label='buy-only gain', color='green')
    plt.plot(range(len(s_money)), s_money, linestyle='solid', label='sell-only gain', color='red')
    plt.plot([0, len(stock_price) - 1], [stock_price[0], stock_price[-1]], linestyle='solid', label='buy and hold',
             color='yellow')

    # plt.plot(range(len(stock_price)), stock_price, linestyle='-', label='stock price', color='gray')
    plt.plot(range(len(stock_price)), [0 for _ in range(len(stock_price))], linestyle=':', color='gray')

    plt.legend()

    if not os.path.exists('../graph'):
        os.makedirs('../graph')

    plt.savefig('../graph/VOL_' + stock_name + '.png', dpi=1000)
    # plt.show()

    plt.close(fig)


def plot_stockprice(stock_name, stock_price):
    fig = plt.figure()

    plt.title(stock_name)
    plt.xlabel('Time')
    plt.ylabel('Price')

    plt.plot(range(len(stock_price)), stock_price, linestyle='-', label='stock price', color='blue')
    # plt.plot(range(len(stock_price)), [0 for _ in range(len(stock_price))], linestyle=':', color='gray')

    plt.legend()

    if not os.path.exists('../graph'):
        os.makedirs('../graph')

    plt.savefig('../graph/VOL_price_' + stock_name + '.png', dpi=1000)
    # plt.show()

    plt.close(fig)
