import pandas as pd
import os

# Use local file paths (no data/ prefix since you're already inside the folder)
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

train_df = pd.read_csv("KDDTrain+.txt", names=columns)
test_df = pd.read_csv("KDDTest+.txt", names=columns)

df = pd.concat([train_df, test_df], ignore_index=True)

# Drop unused columns
df.drop(["difficulty", "land", "is_host_login", "num_outbound_cmds"], axis=1, inplace=True)

# Label: normal -> 0, attack -> 1
df["label"] = df["label"].apply(lambda x: 0 if x.strip() == "normal" else 1)

# Save to current folder as 'nsl_kdd.csv'
df.to_csv("nsl_kdd.csv", index=False)
print("✅ nsl_kdd.csv created successfully with", len(df), "records.")
