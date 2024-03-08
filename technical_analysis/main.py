import numpy as np
from typing import List

from Strategies import gen_signals


class Order:
    def __init__(self, timestamp, bought_at, stop_loss, take_profit, order_type, sold_at=None, is_active: bool = True):
        self.timestamp = timestamp
        self.bought_at = bought_at
        self.sold_at = sold_at
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.order_type = order_type
        self.is_active = is_active

    def __repr__(self) -> str:
        return f"{self.order_type} Position - id {self.timestamp}"


def Backtesting(x: np.array, datos, strategy: List[int], COMMISSION: float = 0.0025, cash: float = 1_000_000):
    com_pasada = 0
    valor_f_port = 0
    positions = []
    closed_positions = []
    STOP_LOSS, TAKE_PROFIT, *strat_args = x
    # print(f"BACKTESTING WITH SL={STOP_LOSS}, TP={TAKE_PROFIT}, ARGS={strat_args}]")
    # print("GENERATING BUY SIGNALS")
    buy_signals = gen_signals(datos, strategy, "BUY", *strat_args)
    # print("GENERATING SELL SIGNALS")
    sell_signals = gen_signals(datos, strategy, "SELL", *strat_args)
    n_rows = len(datos)
    # print("STARTING BACKTESTING LOOP")
    for (i, row), (_, buy_sig_row), (_, sell_sig_row) in zip(datos.iterrows(), buy_signals.iterrows(),
                                                             sell_signals.iterrows()):
        # if i % 1000 == 0:
        #     print(f"Iter {i} / {n_rows}")
        # Close positions
        price = row.Close
        #if i % 9000 == 0:
            #open_pos = len(list(filter(lambda x: x.is_active, positions)))
            #print(f"POSITIONS: {len(positions)} - OPEN POS {open_pos} - Iter {i} / {n_rows}")

        for position in positions:

            j = positions.index(position)
            if position.is_active:

                if position.order_type == "LONG":

                    if price <= position.stop_loss:
                        # Close position, loss:
                        cash += price * (1 - COMMISSION)
                        position.is_active = False
                        position.sold_at = price
                        closed_pos = positions.pop(j)
                        closed_positions.append(closed_pos)
                        # print(f"closing active position - Bought at {position.bought_at}"+
                        #    f"- Sold at {position.sold_at}")

                    elif price >= position.take_profit:
                        # Close position, profit:
                        cash += price * (1 - COMMISSION)
                        position.is_active = False
                        position.sold_at = price
                        closed_pos = positions.pop(j)
                        closed_positions.append(closed_pos)
                        # print(f"closing active position - Bought at {position.bought_at}"+
                        #    f"- Sold at {position.sold_at}")

                if position.order_type == "SHORT":
                    margin_call = cash + (2 * (position.bought_at - row.Close - com_pasada))
                    if cash <= margin_call:
                        if price >= position.stop_loss:

                            # Close position, loss:
                            cash -= price * (1 - COMMISSION)
                            position.is_active = False
                            position.sold_at = price
                            closed_pos = positions.pop(j)
                            closed_positions.append(closed_pos)
                            # print(f"closing active position - Sold at {position.bought_at}"+
                            #    f"- Bought at {position.sold_at}")

                        elif price <= position.take_profit:
                            # Close position, profit:
                            cash -= price * (1 - COMMISSION)
                            position.is_active = False
                            position.sold_at = price
                            closed_pos = positions.pop(j)
                            closed_positions.append(closed_pos)
                            # print(f"closing active position - Sold at {position.bought_at}"+
                            #    f"- Bought at {position.sold_at}")
                        # elif cash >= margin_call:
                        #   print("No hay dinero")

        # buy
        if buy_sig_row.sum() > len(buy_sig_row) // 2:
             print(f"Buy signal @ {row.Close}")
            if cash < row.Close * (1 + COMMISSION):
                print("RAN OUT OF CASH")
                continue
            cash -= row.Close * (1 + COMMISSION)
            com_pasada = COMMISSION

            order = Order(timestamp=row.Timestamp,
                          bought_at=row.Close,
                          stop_loss=row.Close * (1 - STOP_LOSS),
                          take_profit=row.Close * (1 + TAKE_PROFIT),
                          order_type="LONG")

            positions.append(order)

        # sell
        if sell_sig_row.sum() > len(sell_sig_row) // 2:
             print(f"Buy signal @ {row.Close}")
            if cash < row.Close * (1 + COMMISSION):
                continue
            cash += row.Close * (1 - COMMISSION)
            com_pasada = COMMISSION

            order = Order(timestamp=row.Timestamp,
                          bought_at=row.Close,
                          stop_loss=row.Close * (1 + STOP_LOSS),
                          take_profit=row.Close * (1 - TAKE_PROFIT),
                          order_type="SHORT")

            positions.append(order)

    #for position in positions:
       if position.is_active:
            if position.order_type == "LONG":
                valor_f_port += datos.Close[-1] * (1 - COMMISSION)
            else:
                valor_f_port -= datos.Close[-1] * (1 - COMMISSION)
    open_long_positions = list(filter(lambda x: x.order_type == "LONG", positions))
    open_short_positions = list(filter(lambda x: x.order_type == "SHORT", positions))

    valor_f_port = len(open_long_positions)*row.Close + sum([pos.bought_at - row.Close for pos in open_short_positions]) + cash
    print(valor_f_port)

    return valor_f_port
