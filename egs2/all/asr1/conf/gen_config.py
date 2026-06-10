# confidence_estimations = ["uniform", "gaussian", "gumbel", "softmax", "none"]


import glob
import os

import yaml

# update by base config
# conf/auto_regressive
base_config = yaml.safe_load(open("conf/base_config.yaml", "r"))
for i in ["auto_regressive", "ctc", "mask_ctc"]:
    for j in glob.glob(f"conf/{i}/*yaml"):
        config = yaml.safe_load(open(j, "r"))
        for key, value in base_config.items():
            # breakpoint()
            if isinstance(value, dict):
                for _key, _value in value.items():
                    config[key][_key] = _value
            else:
                config[key] = value
        #
        with open(j, "w") as file:
            yaml.dump(config, file, default_flow_style=False)


# file_list = [i for i in glob.glob("conf/condition_mask_ctc/*")]
# for file_path in file_list:
#     for discrete in [True, False]:
#         for gate in ["soft","hard","sum","none"]:
#             for confidence_estimation in ["uniform", "gaussian", "gumbel", "softmax", "predict"]:
#                 #print(confidence_estimation, discrete)
#                 for residual in [True, False]:
#                     if gate == "none":
#                         confidence_estimation = "softmax"
#                     config = yaml.safe_load(open(file_path, 'r'))
#                     if "confidence_estimation" in config["ctc_conf"] or "discrete" in config["ctc_conf"]:
#                         print(file_path)

#                     config["ctc_conf"]["confidence_estimation"] = confidence_estimation
#                     config["ctc_conf"]["discrete"] = discrete
#                     new_path = file_path.replace(".yaml", "").replace("/condition_mask_ctc/", "/proposed/")+f"_{confidence_estimation}"+f"_{gate}"
#                     if residual:
#                         new_path += "_residual"
#                     new_path += "_discret.yaml" if discrete else "_continuous.yaml"
#                     os.makedirs(os.path.dirname(new_path),exist_ok=True)


#                     if gate not in {"hard", "none"}  and  discrete:
#                         #breakpoint()
#                         #continue
#                         pass
#                     else:
#                         with open(new_path, 'w') as file:
#                             yaml.dump(config, file, default_flow_style=False)
