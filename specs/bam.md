# Close All

## High Level Requirements

You are a highly skilled Telegram API expert, Python developer and software architect.

## Mid Level Requirements

Create code/bam.py based on low level requirements.

### Background Information

#### Config File

Revew code/alpaca_config.py.

#### How to cancel positions and orders

Liquidate all positions
python3 code/alpaca.py --account-name [ACCOUNT_NAME] --account [ACCOUNT] --liquidate-all --submit

Cancel all orders
python3 code/alpaca.py --account-name [ACCOUNT_NAME] --account [ACCOUNT] --cancel-all-orders --submit

Where:
  --account-name ACCOUNT_NAME
                        Account name to use (default: Bruce)
  --account ACCOUNT     Account environment to use: paper, live, cash (default: paper)

Always specify account-name and account.

### Standards
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

New script:
- read code/alpaca_config.py: collect all account-name/account combinations for which auto_trade="yes"

- Liquidate all positions for auto_trade="yes": Liquidate accounts in the following order: live, cash, paper.

- Close all positions for auto_trade="yes": Close accounts in the following order: live, cash, paper.