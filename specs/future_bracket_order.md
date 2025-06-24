# Future Bracket Order

## High Level Requirements

You are a highly skilled Python developer.

## Mid Level Requirements

Create a new method in 'code/alpaca.py' called '_futureBracketOrder'.
Mirror the '_bracketOrder' method.
Create all CLI arguments to support the function call from mail.

## Low Level Requirements

In the new method, mirror the call alpaca.submit_order( symbol='AAPL', qty=100, side='buy', type='limit', time_in_force='day', limit_price=150.00, # Your signal buy price order_class='bracket', stop_loss={'stop_price': 145.00}, # Your stop price take_profit={'limit_price': 160.00} # Your target sell price ).  Put a new line after each argument.

Use stop_price as a CLI argument.

Type arguments and create a docstring.

Update README.md with usage instructions.