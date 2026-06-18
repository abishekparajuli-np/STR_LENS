import plotly.graph_objects as go


def _truncate(label: str, max_len: int = 18) -> str:
    """Shortens long account numbers / institution names for the node label
    shown directly on the chart, while the full value is still shown on
    hover via customdata. Long unbroken digit strings (account numbers) are
    the main reason labels overflow their bars in the original chart."""
    label = str(label)
    if len(label) <= max_len:
        return label
    return label[:max_len - 1] + "…"


def draw_transaction_network(sender_account: str, receiver_account: str,
                              sender_bank: str, receiver_bank: str,
                              amount, currency: str = "NPR") -> go.Figure:
    """Create a Sankey diagram visualising a single transaction.

    The diagram shows the flow from the sender's bank to the sender account,
    then to the receiver account and finally to the receiver's bank.
    `amount` is displayed as the link value and on hover, formatted with
    `currency` (defaults to NPR to match the rest of the app, rather than a
    hardcoded $ sign that was misleading for non-USD transactions).

    Parameters
    ----------
    sender_account: str
        Account identifier of the sender.
    receiver_account: str
        Account identifier of the receiver.
    sender_bank: str
        Name of the sender's institution.
    receiver_bank: str
        Name of the receiver's institution.
    amount: Union[int, float, str]
        Transaction amount (numeric or string). It will be cast to float for
        the Sankey value; non-numeric values are shown as 0.
    currency: str
        Currency code shown alongside the amount (default "NPR").

    Returns
    -------
    go.Figure
        Plotly Sankey figure ready to be passed to ``st.plotly_chart``.

    Notes
    -----
    Domestic transactions where sender_bank == receiver_bank (very common in
    AML narrative data) used to break this diagram: building a dict keyed by
    label text collapses the two identical bank-name entries into a single
    index, so the rendered Sankey would show a phantom disconnected node and
    route the money flow incorrectly. Node identity is keyed by role
    (position in the flow) rather than by label text, so two nodes can share
    a display label without being treated as the same node. The same
    protection applies if the sender and receiver account numbers happen to
    be identical (e.g. test/dummy data, or a self-transfer).
    """
    try:
        amt_value = float(amount)
    except (TypeError, ValueError):
        amt_value = 0.0

    sender_account = str(sender_account) if sender_account else "Unknown account"
    receiver_account = str(receiver_account) if receiver_account else "Unknown account"
    sender_bank = str(sender_bank) if sender_bank else "Unknown institution"
    receiver_bank = str(receiver_bank) if receiver_bank else "Unknown institution"

    # Keyed by role (list position), not by label text -- see Notes above.
    SENDER_BANK, SENDER_ACC, RECEIVER_ACC, RECEIVER_BANK = 0, 1, 2, 3

    full_labels = [sender_bank, sender_account, receiver_account, receiver_bank]
    display_labels = [_truncate(lbl) for lbl in full_labels]
    role_names = ["Sender's Institution", "Sender Account", "Receiver Account", "Receiver's Institution"]

    node_colors = ["#3B82F6", "#1E3A8A", "#0E7490", "#F59E0B"]
    node_hover = [
        f"<b>{role}</b><br>{full}"
        for role, full in zip(role_names, full_labels)
    ]

    source = [SENDER_BANK, SENDER_ACC, RECEIVER_ACC]
    target = [SENDER_ACC, RECEIVER_ACC, RECEIVER_BANK]
    value = [amt_value, amt_value, amt_value]
    link_labels = [
        f"{currency} {amt_value:,.2f}",
        f"{currency} {amt_value:,.2f}",
        f"{currency} {amt_value:,.2f}",
    ]
    link_hover = [
        f"<b>Deposit / Origination</b><br>{sender_bank} \u2192 {sender_account}<br>{currency} {amt_value:,.2f}",
        f"<b>Transfer</b><br>{sender_account} \u2192 {receiver_account}<br>{currency} {amt_value:,.2f}",
        f"<b>Receipt / Settlement</b><br>{receiver_account} \u2192 {receiver_bank}<br>{currency} {amt_value:,.2f}",
    ]
    link_colors = ["rgba(59,130,246,0.35)", "rgba(14,116,144,0.40)", "rgba(245,158,11,0.35)"]

    sankey = go.Sankey(
        arrangement="snap",
        node=dict(
            pad=24,
            thickness=24,
            line=dict(color="rgba(255,255,255,0.3)", width=1),
            label=display_labels,
            color=node_colors,
            customdata=node_hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            label=link_labels,
            color=link_colors,
            customdata=link_hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        textfont=dict(color="#E2E8F0", size=13, family="Inter, sans-serif"),
    )

    fig = go.Figure(sankey)
    fig.update_layout(
        title=dict(
            text=f"Transaction Flow \u2014 {currency} {amt_value:,.2f}",
            font=dict(size=16, color="#E2E8F0", family="Inter, sans-serif"),
        ),
        font=dict(size=12, color="#E2E8F0", family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        height=320,
    )
    return fig