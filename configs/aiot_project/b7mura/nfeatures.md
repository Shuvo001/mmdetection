
##Neck

###PAFPN

|配置|W PAFPN|WO PAFPN(FPN)|
|---|---|---|
|rcnn_pafpn.py/rcnn_yoloxv2_snun.py|0.7905|0.7878|
|rcnn_scale_pafpn.py/rcnn_yoloxv2_scale.py|0.8095|0.797|


###PAFPN short cut

|配置|W PAFPN|W PAFPN(short cut=True)|
|---|---|---|
|rcnn_pafpn.py/rcnn_ps.py|0.7905|0.771|
|rcnn_nassigner.py/rcnn_nas.py|0.772|0.781|

###input normal

|配置|W input normal|WO inputnormal|
|---|---|---|
|rcnn_yoloxv2_sn.py/rcnn_yoloxv2_scale.py|0.771|0.797|

###WS+GN

|配置|W WS+GN|WO WS+GN|
|---|---|---|
|rcnn_wspafpn.py/rcnn_pafpn.py|0.768|0.7905|
|rcnn_head_ws.py/rcnn_yoloxv2_snun.py|0.791|0.787|
|rcnn_ms_ws_ta.py/rcnn_ms_ta.py|0.779|0.791|
|rcnn_scale_na_pafpn_ws_ta.py/|0.78|0.0.799|

###backbone WS+GN

|配置|W WS|WO WS|
|---|---|---|
|rcnn_ws.py(bs=4)/rcnn_yoloxv2_snun.py|0.774|0.787|

###resample

|配置|W resample|WO resample|
|---|---|---|
|rcnn_yoloxv2_huges_redbc.py/rcnn_yoloxv2_huges_dbc.py|0.772|0.776|

###neck norm
|配置|WO norm|W BN|W EvoNormS0|W GN|
|---|---|---|---|---|
|config|0.772|0.768|0.782|0.781|

config:
- rcnn_yoloxv2_huges_redbc.py
- rcnn_yoloxv2_huges_redbcbn.py
- rcnn_yoloxv2_huges_redbcen.py
- rcnn_yoloxv2_huges_redbcgn.py

###stem

|配置|MultiBranchStemS12X|MultiBranchStemSA12X|
|---|---|---|
|config0|0.774|0.790/0.771|
|config|0.796|0.797|

config0:
- rcnn_yoloxv2_huge.py
- rcnn_yoloxv2_huges_bn.py/rcnn_yoloxv2_hugesa.py

config1:
- rcnn_scales.py
- rcnn_yoloxv2_scale.py

###Head FC norm

|配置|WO GN|W norm|
|---|---|---|
|rcnn_yoloxv2_hugesa.py/rcnn_yoloxv2_hugesaw.py|0.771|0.768|
|rcnn_yoloxv2_ssnn.py/rcnn_yoloxv2_ssn.py|0.766|0.789|
|rcnn_yoloxv2_snun.py/rcnn_yoloxv2_snu.py|0.789|0.772|


###Head

|配置|WShared4Conv2FCBBoxHead|WConvFCSBBoxHead|
|---|---|---|
|config|0.768|0.753|

config:

- rcnn_yoloxv2_hugesaw.py
- rcnn_yoloxv2_hugesaws.py

|配置|WShared4Conv2FCBBoxHead|WConvFCSBBoxHead|
|---|---|---|
|config|0.771|0.789|

config:

- rcnn_yoloxv2_sn.py
- rcnn_yoloxv2_ssn.py

###BN momentum

|配置|0.1|0.03|
|---|---|---|
|config|0.771|0.773|

config:
- rcnn_yoloxv2_hugesa.py
- rcnn_yoloxv2_huges_bnm.py

###Loss scale

|配置|W loss scale|WO loss scale|
|---|---|---|
|rcnn_yoloxv2_scale.py/rcnn_yoloxv2_hugesaw.py|0.797|0.768|

### dropblock enable function

|配置|no enable function|enable function|
|---|---|---|
|rcnn_yoloxv2_sn.py/rcnn_yoloxv2_snd.py|0.771|0.780|

###使用OK样本


|配置|no OK|OK|
|---|---|---|
|config|0.771|0.772|

config:
- rcnn_yoloxv2_sn.py
- rcnn_yoloxv2_snu.py


|配置|OK|OK resample|
|---|---|---|
|config|0.789|0.776|

config:
- rcnn_yoloxv2_snun.py
- rcnn_ren.py


###rsb预训练权重

|配置|W rsb|WO rsb|
|---|---|---|
|rcnn_yoloxv2_1wsnun.py/rcnn_yoloxv2_snun.py|0.764|0.789|

###backbone

|配置|R50|S50|
|---|---|---|
|config|0.789|0.782|

config:
- rcnn_yoloxv2_snun.py
- rcnn_s50.py

|配置|R50|CSPResnet50|
|---|---|---|
|config0|0.791|0.796|
|config1|0.791|0.777|

config0:
- rcnn_pafpn.py
- rcnn_cr.py

config0:
- rcnn_hr.py
- rcnn_hr_cr.py

###SimOTA vs SimOTA(min bbox size=50)

|配置|SimOTA|SimOTA(ms=50)|
|---|---|---|
|rcnn_yoloxv2_scale.py/rcnn_scale_na.py|0.797|0.799|
|rcnn_pafpn.py/rcnn_nassigner.py|0.791|0.773|
|rcnn_scale_pafpn.py/rcnn_scale_na_pafpn.py|0.8095|0.782|

###train strategy

|配置|ref|先训练stem和head再训练所有权重|
|---|---|---|
|rcnn_scale_na_pafpn.py|0.782|0.813|
|rcnn_ms.py(sr2)|0.790|0.810|


|配置|ref|训练到一半使用hook训练所有权重|
|---|---|---|
|rcnn_scale_na_pafpn.py/rcnn_scale_na_pafpn_ta.py|0.782|0.799|
|rcnn_ms.py/rcnn_ms_ta.py|0.799|0.791|

|配置|ref|使用两个gpu训练|
|---|---|---|
|rcnn_scale_na_pafpn.py|0.782|0.781|

|配置|ref|先训练stem和head再用大batchsize训练head|
|---|---|---|
|rcnn_scale_na_pafpn.py|0.782|0.802|
|rcnn_ws.py/rcnn_wsh|0.790|0.795|

|配置|训练到一半使用hook训练所有权重|不训练BN|
|---|---|---|
|rcnn_scale_na_pafpn_ta.py/rcnn_scale_na_pafpn_tabn.py|0.799|0.791|

|配置|ref|直接训练所有参数|
|---|---|---|
|rcnn_scale_na_pafpn.py/rcnn_scale_na_pafpna.py|0.782|0.81|

|配置|ref|训练100000步|
|---|---|---|
|rcnn_scale_na_pafpn.py/rcnn_scale_na_pafpnl.py|0.782|0.795|

###multi scale rcnn
|配置|w multi scale rcnn|wo multi scale rcnn|
|---|---|---|
|rcnn_ms.py/rcnn_scale_na_pafpn.py|0.799|0.782|
|rcnn_ms_ta.py/rcnn_scale_na_pafpn_ta.py|0.791|0.799|

###different dataset
|配置|s1|s2|
|---|---|---|
|rcnn_ms.py|0.799|0.790|
