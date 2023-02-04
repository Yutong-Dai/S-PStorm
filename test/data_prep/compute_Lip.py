from scipy.io import savemat

for db in ["a9a", "avazu-app.tr", "covtype", "kdda", "new20", "phishing", "rcv1", "real-sim", "url_combined", "w8a"]:
    Lip_path = f"./Lip/Lip-{db}.mat"
    savemat(Lip_path, {"L": 0.25})
