def calculate_clv(tenure, monthly_charges):
    try:
        tenure = float(tenure)
        monthly_charges = float(monthly_charges)
    except:
        return 'Low'
    clv = tenure * monthly_charges
    if clv > 1500:
        return 'High'
    elif clv > 700:
        return 'Medium'
    else:
        return 'Low'
