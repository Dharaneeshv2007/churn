def get_recommendation(risk_level):
    if risk_level == 'High':
        return 'Offer discount'
    elif risk_level == 'Medium':
        return 'Engagement offer'
    else:
        return 'No action'
