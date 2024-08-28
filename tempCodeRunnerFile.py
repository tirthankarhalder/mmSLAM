
# defined global path
today = datetime.date.today()
date_string = today.strftime('%Y-%m-%d-%H-%M-%S')
filepath = "./datasets/"
filepath+=date_string
Path(filepath).mkdir(parents=True,exist_ok=True)
