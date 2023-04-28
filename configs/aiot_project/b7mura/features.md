
##Neck

###PAFPN

|配置|W PAFPN|WO PAFPN(FPN)|
|---|---|---|
|rcnn_pafpn.py/rcnn_yoloxv2_snun.py|0.640|0.620|


###input normal

|配置|W input normal|WO inputnormal|
|---|---|---|
|rcnn_yoloxv2_sn.py/rcnn_yoloxv2_scale.py|0.614|0.634|

###WS+GN

|配置|W input normal|WO inputnormal|
|---|---|---|
|rcnn_wspafpn.py/rcnn_pafpn.py|0.610|0.640|
|rcnn_head_ws.py/rcnn_yoloxv2_snun.py|0.621|0.620|

###backbone WS+GN

|配置|W WS|WO WS|
|---|---|---|
|rcnn_ws.py(bs=4)/rcnn_yoloxv2_snun.py|0.585|0.620|

###resample

|配置|W resample|WO resample|
|---|---|---|
|rcnn_yoloxv2_huges_redbc.py/rcnn_yoloxv2_huges_dbc.py|0.6|0.581|

###neck norm
|配置|WO norm|W BN|W EvoNormS0|W GN|
|---|---|---|---|---|
|config|0.6|0.605|0.593|0.624|

config:
- rcnn_yoloxv2_huges_redbc.py
- rcnn_yoloxv2_huges_redbcbn.py
- rcnn_yoloxv2_huges_redbcen.py
- rcnn_yoloxv2_huges_redbcgn.py

###stem

|配置|MultiBranchStemS12X|MultiBranchStemSA12X|
|---|---|---|
|config|0.598|0.605/0.582|

config:
- rcnn_yoloxv2_huge.py
- rcnn_yoloxv2_huges_bn.py/rcnn_yoloxv2_hugesa.py

###Head FC norm

|配置|WO GN|W norm|
|---|---|---|
|rcnn_yoloxv2_hugesa.py/rcnn_yoloxv2_hugesaw.py|0.582|0.6|
|rcnn_yoloxv2_ssnn.py/rcnn_yoloxv2_ssn.py|0.607|0.605|
|rcnn_yoloxv2_snun.py/rcnn_yoloxv2_snu.py|0.620|0.623|


###Head

|配置|WShared4Conv2FCBBoxHead|WConvFCSBBoxHead|
|---|---|---|
|config|0.6|0.568|

config:

- rcnn_yoloxv2_hugesaw.py
- rcnn_yoloxv2_hugesaws.py

|配置|WShared4Conv2FCBBoxHead|WConvFCSBBoxHead|
|---|---|---|
|config|0.614|0.605|

config:

- rcnn_yoloxv2_sn.py
- rcnn_yoloxv2_ssn.py

###BN momentum

|配置|0.1|0.03|
|---|---|---|
|config|0.582|0.592|

config:
- rcnn_yoloxv2_hugesa.py
- rcnn_yoloxv2_huges_bnm.py

###Loss scale

|配置|W loss scale|WO loss scale|
|---|---|---|
|rcnn_yoloxv2_scale.py/rcnn_yoloxv2_hugesaw.py|0.634|0.602|

### dropblock enable function

|配置|no enable function|enable function|
|---|---|---|
|rcnn_yoloxv2_sn.py/rcnn_yoloxv2_snd.py|0.614|0.612|

###使用OK样本


|配置|no OK|OK|
|---|---|---|
|config|0.614|0.623|

config:
- rcnn_yoloxv2_sn.py
- rcnn_yoloxv2_snu.py


|配置|OK|OK resample|
|---|---|---|
|config|0.620|0.614|

config:
- rcnn_yoloxv2_snun.py
- rcnn_ren.py


###rsb预训练权重

|配置|W rsb|WO rsb|
|---|---|---|
|rcnn_yoloxv2_1wsnun.py/rcnn_yoloxv2_snun.py|0.576|0.620|

###backbone

|配置|R50|S50|
|---|---|---|
|config|0.620|0.599|

config:
- rcnn_yoloxv2_snun.py
- rcnn_s50.py
