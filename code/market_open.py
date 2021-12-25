import alpaca_trade_api as tradeapi

api = tradeapi.REST()

# Check if the market is open now.
clock = api.get_clock()
print('The market is {}'.format('open.' if clock.is_open else 'closed.'))

# Check when the market was open on Dec. 1, 2018
date = '2018-12-01'
calendar = api.get_calendar(start=date, end=date)[0]
print('The market opened at {} and closed at {} on {}.'.format(
    calendar.open,
    calendar.close,
    date
))

# Check when the market was open in Dec. 2021
startDate = '2021-12-01'
stopDate  = '2021-12-31'
calendars = api.get_calendar(start=startDate, end=stopDate)

for calendar in calendars:
    print(f'calendar: {calendar}')
    print(f'The market opened at {calendar.open} and closed at {calendar.close} on {calendar.date}.')
