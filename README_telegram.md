# Telegram Commands

## Alpaca commands

```text
<trigger-word> <alpaca args>
```

```text
<trigger-word> -buy-trailing -symbol CONI  -amount 1000 -trailing-percent 7.5
```

## Plotting

Date defaults to current date.

```text
plot -plot -symbol WLDS -date 2025-09-10
```

## Bam

Close all positions and orders in ALL accounts where auto_trade="yes"

```text
bam
```