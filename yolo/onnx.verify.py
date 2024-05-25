import onnx

model = onnx.load("yolov8n_blood_cell_detection.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))


# graph main_graph (
#   %input[FLOAT, batch_sizex3x640x640]
# ) initializers (
#   %model.22.cv2.0.2.weight[FLOAT, 64x64x1x1]
#   %model.22.cv2.0.2.bias[FLOAT, 64]
#   %model.22.cv2.1.2.weight[FLOAT, 64x64x1x1]
#   %model.22.cv2.1.2.bias[FLOAT, 64]
#   %model.22.cv2.2.2.weight[FLOAT, 64x64x1x1]
#   %model.22.cv2.2.2.bias[FLOAT, 64]
#   %model.22.cv3.0.2.weight[FLOAT, 3x64x1x1]
#   %model.22.cv3.0.2.bias[FLOAT, 3]
#   %model.22.cv3.1.2.weight[FLOAT, 3x64x1x1]
#   %model.22.cv3.1.2.bias[FLOAT, 3]
#   %model.22.cv3.2.2.weight[FLOAT, 3x64x1x1]
#   %model.22.cv3.2.2.bias[FLOAT, 3]
#   %model.22.dfl.conv.weight[FLOAT, 1x16x1x1]
#   %onnx::Conv_953[FLOAT, 16x3x3x3]
#   %onnx::Conv_954[FLOAT, 16]
#   %onnx::Conv_956[FLOAT, 32x16x3x3]
#   %onnx::Conv_957[FLOAT, 32]
#   %onnx::Conv_959[FLOAT, 32x32x1x1]
#   %onnx::Conv_960[FLOAT, 32]
#   %onnx::Conv_962[FLOAT, 16x16x3x3]
#   %onnx::Conv_963[FLOAT, 16]
#   %onnx::Conv_965[FLOAT, 16x16x3x3]
#   %onnx::Conv_966[FLOAT, 16]
#   %onnx::Conv_968[FLOAT, 32x48x1x1]
#   %onnx::Conv_969[FLOAT, 32]
#   %onnx::Conv_971[FLOAT, 64x32x3x3]
#   %onnx::Conv_972[FLOAT, 64]
#   %onnx::Conv_974[FLOAT, 64x64x1x1]
#   %onnx::Conv_975[FLOAT, 64]
#   %onnx::Conv_977[FLOAT, 32x32x3x3]
#   %onnx::Conv_978[FLOAT, 32]
#   %onnx::Conv_980[FLOAT, 32x32x3x3]
#   %onnx::Conv_981[FLOAT, 32]
#   %onnx::Conv_983[FLOAT, 32x32x3x3]
#   %onnx::Conv_984[FLOAT, 32]
#   %onnx::Conv_986[FLOAT, 32x32x3x3]
#   %onnx::Conv_987[FLOAT, 32]
#   %onnx::Conv_989[FLOAT, 64x128x1x1]
#   %onnx::Conv_990[FLOAT, 64]
#   %onnx::Conv_992[FLOAT, 128x64x3x3]
#   %onnx::Conv_993[FLOAT, 128]
#   %onnx::Conv_995[FLOAT, 128x128x1x1]
#   %onnx::Conv_996[FLOAT, 128]
#   %onnx::Conv_998[FLOAT, 64x64x3x3]
#   %onnx::Conv_999[FLOAT, 64]
#   %onnx::Conv_1001[FLOAT, 64x64x3x3]
#   %onnx::Conv_1002[FLOAT, 64]
#   %onnx::Conv_1004[FLOAT, 64x64x3x3]
#   %onnx::Conv_1005[FLOAT, 64]
#   %onnx::Conv_1007[FLOAT, 64x64x3x3]
#   %onnx::Conv_1008[FLOAT, 64]
#   %onnx::Conv_1010[FLOAT, 128x256x1x1]
#   %onnx::Conv_1011[FLOAT, 128]
#   %onnx::Conv_1013[FLOAT, 256x128x3x3]
#   %onnx::Conv_1014[FLOAT, 256]
#   %onnx::Conv_1016[FLOAT, 256x256x1x1]
#   %onnx::Conv_1017[FLOAT, 256]
#   %onnx::Conv_1019[FLOAT, 128x128x3x3]
#   %onnx::Conv_1020[FLOAT, 128]
#   %onnx::Conv_1022[FLOAT, 128x128x3x3]
#   %onnx::Conv_1023[FLOAT, 128]
#   %onnx::Conv_1025[FLOAT, 256x384x1x1]
#   %onnx::Conv_1026[FLOAT, 256]
#   %onnx::Conv_1028[FLOAT, 128x256x1x1]
#   %onnx::Conv_1029[FLOAT, 128]
#   %onnx::Conv_1031[FLOAT, 256x512x1x1]
#   %onnx::Conv_1032[FLOAT, 256]
#   %onnx::Conv_1034[FLOAT, 128x384x1x1]
#   %onnx::Conv_1035[FLOAT, 128]
#   %onnx::Conv_1037[FLOAT, 64x64x3x3]
#   %onnx::Conv_1038[FLOAT, 64]
#   %onnx::Conv_1040[FLOAT, 64x64x3x3]
#   %onnx::Conv_1041[FLOAT, 64]
#   %onnx::Conv_1043[FLOAT, 128x192x1x1]
#   %onnx::Conv_1044[FLOAT, 128]
#   %onnx::Conv_1046[FLOAT, 64x192x1x1]
#   %onnx::Conv_1047[FLOAT, 64]
#   %onnx::Conv_1049[FLOAT, 32x32x3x3]
#   %onnx::Conv_1050[FLOAT, 32]
#   %onnx::Conv_1052[FLOAT, 32x32x3x3]
#   %onnx::Conv_1053[FLOAT, 32]
#   %onnx::Conv_1055[FLOAT, 64x96x1x1]
#   %onnx::Conv_1056[FLOAT, 64]
#   %onnx::Conv_1058[FLOAT, 64x64x3x3]
#   %onnx::Conv_1059[FLOAT, 64]
#   %onnx::Conv_1061[FLOAT, 128x192x1x1]
#   %onnx::Conv_1062[FLOAT, 128]
#   %onnx::Conv_1064[FLOAT, 64x64x3x3]
#   %onnx::Conv_1065[FLOAT, 64]
#   %onnx::Conv_1067[FLOAT, 64x64x3x3]
#   %onnx::Conv_1068[FLOAT, 64]
#   %onnx::Conv_1070[FLOAT, 128x192x1x1]
#   %onnx::Conv_1071[FLOAT, 128]
#   %onnx::Conv_1073[FLOAT, 128x128x3x3]
#   %onnx::Conv_1074[FLOAT, 128]
#   %onnx::Conv_1076[FLOAT, 256x384x1x1]
#   %onnx::Conv_1077[FLOAT, 256]
#   %onnx::Conv_1079[FLOAT, 128x128x3x3]
#   %onnx::Conv_1080[FLOAT, 128]
#   %onnx::Conv_1082[FLOAT, 128x128x3x3]
#   %onnx::Conv_1083[FLOAT, 128]
#   %onnx::Conv_1085[FLOAT, 256x384x1x1]
#   %onnx::Conv_1086[FLOAT, 256]
#   %onnx::Conv_1088[FLOAT, 64x64x3x3]
#   %onnx::Conv_1089[FLOAT, 64]
#   %onnx::Conv_1091[FLOAT, 64x64x3x3]
#   %onnx::Conv_1092[FLOAT, 64]
#   %onnx::Conv_1094[FLOAT, 64x64x3x3]
#   %onnx::Conv_1095[FLOAT, 64]
#   %onnx::Conv_1097[FLOAT, 64x64x3x3]
#   %onnx::Conv_1098[FLOAT, 64]
#   %onnx::Conv_1100[FLOAT, 64x128x3x3]
#   %onnx::Conv_1101[FLOAT, 64]
#   %onnx::Conv_1103[FLOAT, 64x64x3x3]
#   %onnx::Conv_1104[FLOAT, 64]
#   %onnx::Conv_1106[FLOAT, 64x128x3x3]
#   %onnx::Conv_1107[FLOAT, 64]
#   %onnx::Conv_1109[FLOAT, 64x64x3x3]
#   %onnx::Conv_1110[FLOAT, 64]
#   %onnx::Conv_1112[FLOAT, 64x256x3x3]
#   %onnx::Conv_1113[FLOAT, 64]
#   %onnx::Conv_1115[FLOAT, 64x64x3x3]
#   %onnx::Conv_1116[FLOAT, 64]
#   %onnx::Conv_1118[FLOAT, 64x256x3x3]
#   %onnx::Conv_1119[FLOAT, 64]
#   %onnx::Conv_1121[FLOAT, 64x64x3x3]
#   %onnx::Conv_1122[FLOAT, 64]
# ) {
#   %/model.0/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]](%input, %onnx::Conv_953, %onnx::Conv_954)
#   %/model.0/act/Sigmoid_output_0 = Sigmoid(%/model.0/conv/Conv_output_0)
#   %/model.0/act/Mul_output_0 = Mul(%/model.0/conv/Conv_output_0, %/model.0/act/Sigmoid_output_0)
#   %/model.1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]](%/model.0/act/Mul_output_0, %onnx::Conv_956, %onnx::Conv_957)
#   %/model.1/act/Sigmoid_output_0 = Sigmoid(%/model.1/conv/Conv_output_0)
#   %/model.1/act/Mul_output_0 = Mul(%/model.1/conv/Conv_output_0, %/model.1/act/Sigmoid_output_0)
#   %/model.2/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.1/act/Mul_output_0, %onnx::Conv_959, %onnx::Conv_960)
#   %/model.2/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.2/cv1/conv/Conv_output_0)
#   %/model.2/cv1/act/Mul_output_0 = Mul(%/model.2/cv1/conv/Conv_output_0, %/model.2/cv1/act/Sigmoid_output_0)
#   %/model.2/Shape_output_0 = Shape(%/model.2/cv1/act/Mul_output_0)
#   %/model.2/Constant_output_0 = Constant[value = <Tensor>]()
#   %/model.2/Gather_output_0 = Gather[axis = 0](%/model.2/Shape_output_0, %/model.2/Constant_output_0)
#   %/model.2/Constant_1_output_0 = Constant[value = <Tensor>]()
#   %/model.2/Constant_2_output_0 = Constant[value = <Tensor>]()
#   %/model.2/Add_output_0 = Add(%/model.2/Gather_output_0, %/model.2/Constant_2_output_0)
#   %/model.2/Constant_3_output_0 = Constant[value = <Tensor>]()
#   %/model.2/Div_output_0 = Div(%/model.2/Add_output_0, %/model.2/Constant_3_output_0)
#   %/model.2/Constant_4_output_0 = Constant[value = <Tensor>]()
#   %/model.2/Mul_output_0 = Mul(%/model.2/Div_output_0, %/model.2/Constant_4_output_0)
#   %/model.2/Slice_output_0 = Slice(%/model.2/cv1/act/Mul_output_0, %/model.2/Constant_1_output_0, %/model.2/Mul_output_0, %/model.2/Constant_output_0)
#   %/model.2/Constant_5_output_0 = Constant[value = <Tensor>]()
#   %/model.2/Mul_1_output_0 = Mul(%/model.2/Div_output_0, %/model.2/Constant_5_output_0)
#   %/model.2/Slice_1_output_0 = Slice(%/model.2/cv1/act/Mul_output_0, %/model.2/Mul_output_0, %/model.2/Mul_1_output_0, %/model.2/Constant_output_0)
#   %/model.2/m.0/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.2/Slice_1_output_0, %onnx::Conv_962, %onnx::Conv_963)
#   %/model.2/m.0/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.2/m.0/cv1/conv/Conv_output_0)
#   %/model.2/m.0/cv1/act/Mul_output_0 = Mul(%/model.2/m.0/cv1/conv/Conv_output_0, %/model.2/m.0/cv1/act/Sigmoid_output_0)
#   %/model.2/m.0/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.2/m.0/cv1/act/Mul_output_0, %onnx::Conv_965, %onnx::Conv_966)
#   %/model.2/m.0/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.2/m.0/cv2/conv/Conv_output_0)
#   %/model.2/m.0/cv2/act/Mul_output_0 = Mul(%/model.2/m.0/cv2/conv/Conv_output_0, %/model.2/m.0/cv2/act/Sigmoid_output_0)
#   %/model.2/m.0/Add_output_0 = Add(%/model.2/Slice_1_output_0, %/model.2/m.0/cv2/act/Mul_output_0)
#   %/model.2/Concat_output_0 = Concat[axis = 1](%/model.2/Slice_output_0, %/model.2/Slice_1_output_0, %/model.2/m.0/Add_output_0)
#   %/model.2/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.2/Concat_output_0, %onnx::Conv_968, %onnx::Conv_969)
#   %/model.2/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.2/cv2/conv/Conv_output_0)
#   %/model.2/cv2/act/Mul_output_0 = Mul(%/model.2/cv2/conv/Conv_output_0, %/model.2/cv2/act/Sigmoid_output_0)
#   %/model.3/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]](%/model.2/cv2/act/Mul_output_0, %onnx::Conv_971, %onnx::Conv_972)
#   %/model.3/act/Sigmoid_output_0 = Sigmoid(%/model.3/conv/Conv_output_0)
#   %/model.3/act/Mul_output_0 = Mul(%/model.3/conv/Conv_output_0, %/model.3/act/Sigmoid_output_0)
#   %/model.4/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.3/act/Mul_output_0, %onnx::Conv_974, %onnx::Conv_975)
#   %/model.4/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.4/cv1/conv/Conv_output_0)
#   %/model.4/cv1/act/Mul_output_0 = Mul(%/model.4/cv1/conv/Conv_output_0, %/model.4/cv1/act/Sigmoid_output_0)
#   %/model.4/Shape_output_0 = Shape(%/model.4/cv1/act/Mul_output_0)
#   %/model.4/Constant_output_0 = Constant[value = <Tensor>]()
#   %/model.4/Gather_output_0 = Gather[axis = 0](%/model.4/Shape_output_0, %/model.4/Constant_output_0)
#   %/model.4/Constant_1_output_0 = Constant[value = <Tensor>]()
#   %/model.4/Constant_2_output_0 = Constant[value = <Tensor>]()
#   %/model.4/Add_output_0 = Add(%/model.4/Gather_output_0, %/model.4/Constant_2_output_0)
#   %/model.4/Constant_3_output_0 = Constant[value = <Tensor>]()
#   %/model.4/Div_output_0 = Div(%/model.4/Add_output_0, %/model.4/Constant_3_output_0)
#   %/model.4/Constant_4_output_0 = Constant[value = <Tensor>]()
#   %/model.4/Mul_output_0 = Mul(%/model.4/Div_output_0, %/model.4/Constant_4_output_0)
#   %/model.4/Slice_output_0 = Slice(%/model.4/cv1/act/Mul_output_0, %/model.4/Constant_1_output_0, %/model.4/Mul_output_0, %/model.4/Constant_output_0)
#   %/model.4/Constant_5_output_0 = Constant[value = <Tensor>]()
#   %/model.4/Mul_1_output_0 = Mul(%/model.4/Div_output_0, %/model.4/Constant_5_output_0)
#   %/model.4/Slice_1_output_0 = Slice(%/model.4/cv1/act/Mul_output_0, %/model.4/Mul_output_0, %/model.4/Mul_1_output_0, %/model.4/Constant_output_0)
#   %/model.4/m.0/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.4/Slice_1_output_0, %onnx::Conv_977, %onnx::Conv_978)
#   %/model.4/m.0/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.4/m.0/cv1/conv/Conv_output_0)
#   %/model.4/m.0/cv1/act/Mul_output_0 = Mul(%/model.4/m.0/cv1/conv/Conv_output_0, %/model.4/m.0/cv1/act/Sigmoid_output_0)
#   %/model.4/m.0/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.4/m.0/cv1/act/Mul_output_0, %onnx::Conv_980, %onnx::Conv_981)
#   %/model.4/m.0/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.4/m.0/cv2/conv/Conv_output_0)
#   %/model.4/m.0/cv2/act/Mul_output_0 = Mul(%/model.4/m.0/cv2/conv/Conv_output_0, %/model.4/m.0/cv2/act/Sigmoid_output_0)
#   %/model.4/m.0/Add_output_0 = Add(%/model.4/Slice_1_output_0, %/model.4/m.0/cv2/act/Mul_output_0)
#   %/model.4/m.1/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.4/m.0/Add_output_0, %onnx::Conv_983, %onnx::Conv_984)
#   %/model.4/m.1/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.4/m.1/cv1/conv/Conv_output_0)
#   %/model.4/m.1/cv1/act/Mul_output_0 = Mul(%/model.4/m.1/cv1/conv/Conv_output_0, %/model.4/m.1/cv1/act/Sigmoid_output_0)
#   %/model.4/m.1/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.4/m.1/cv1/act/Mul_output_0, %onnx::Conv_986, %onnx::Conv_987)
#   %/model.4/m.1/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.4/m.1/cv2/conv/Conv_output_0)
#   %/model.4/m.1/cv2/act/Mul_output_0 = Mul(%/model.4/m.1/cv2/conv/Conv_output_0, %/model.4/m.1/cv2/act/Sigmoid_output_0)
#   %/model.4/m.1/Add_output_0 = Add(%/model.4/m.0/Add_output_0, %/model.4/m.1/cv2/act/Mul_output_0)
#   %/model.4/Concat_output_0 = Concat[axis = 1](%/model.4/Slice_output_0, %/model.4/Slice_1_output_0, %/model.4/m.0/Add_output_0, %/model.4/m.1/Add_output_0)
#   %/model.4/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.4/Concat_output_0, %onnx::Conv_989, %onnx::Conv_990)
#   %/model.4/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.4/cv2/conv/Conv_output_0)
#   %/model.4/cv2/act/Mul_output_0 = Mul(%/model.4/cv2/conv/Conv_output_0, %/model.4/cv2/act/Sigmoid_output_0)
#   %/model.5/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]](%/model.4/cv2/act/Mul_output_0, %onnx::Conv_992, %onnx::Conv_993)
#   %/model.5/act/Sigmoid_output_0 = Sigmoid(%/model.5/conv/Conv_output_0)
#   %/model.5/act/Mul_output_0 = Mul(%/model.5/conv/Conv_output_0, %/model.5/act/Sigmoid_output_0)
#   %/model.6/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.5/act/Mul_output_0, %onnx::Conv_995, %onnx::Conv_996)
#   %/model.6/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.6/cv1/conv/Conv_output_0)
#   %/model.6/cv1/act/Mul_output_0 = Mul(%/model.6/cv1/conv/Conv_output_0, %/model.6/cv1/act/Sigmoid_output_0)
#   %/model.6/Shape_output_0 = Shape(%/model.6/cv1/act/Mul_output_0)
#   %/model.6/Constant_output_0 = Constant[value = <Tensor>]()
#   %/model.6/Gather_output_0 = Gather[axis = 0](%/model.6/Shape_output_0, %/model.6/Constant_output_0)
#   %/model.6/Constant_1_output_0 = Constant[value = <Tensor>]()
#   %/model.6/Constant_2_output_0 = Constant[value = <Tensor>]()
#   %/model.6/Add_output_0 = Add(%/model.6/Gather_output_0, %/model.6/Constant_2_output_0)
#   %/model.6/Constant_3_output_0 = Constant[value = <Tensor>]()
#   %/model.6/Div_output_0 = Div(%/model.6/Add_output_0, %/model.6/Constant_3_output_0)
#   %/model.6/Constant_4_output_0 = Constant[value = <Tensor>]()
#   %/model.6/Mul_output_0 = Mul(%/model.6/Div_output_0, %/model.6/Constant_4_output_0)
#   %/model.6/Slice_output_0 = Slice(%/model.6/cv1/act/Mul_output_0, %/model.6/Constant_1_output_0, %/model.6/Mul_output_0, %/model.6/Constant_output_0)
#   %/model.6/Constant_5_output_0 = Constant[value = <Tensor>]()
#   %/model.6/Mul_1_output_0 = Mul(%/model.6/Div_output_0, %/model.6/Constant_5_output_0)
#   %/model.6/Slice_1_output_0 = Slice(%/model.6/cv1/act/Mul_output_0, %/model.6/Mul_output_0, %/model.6/Mul_1_output_0, %/model.6/Constant_output_0)
#   %/model.6/m.0/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.6/Slice_1_output_0, %onnx::Conv_998, %onnx::Conv_999)
#   %/model.6/m.0/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.6/m.0/cv1/conv/Conv_output_0)
#   %/model.6/m.0/cv1/act/Mul_output_0 = Mul(%/model.6/m.0/cv1/conv/Conv_output_0, %/model.6/m.0/cv1/act/Sigmoid_output_0)
#   %/model.6/m.0/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.6/m.0/cv1/act/Mul_output_0, %onnx::Conv_1001, %onnx::Conv_1002)
#   %/model.6/m.0/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.6/m.0/cv2/conv/Conv_output_0)
#   %/model.6/m.0/cv2/act/Mul_output_0 = Mul(%/model.6/m.0/cv2/conv/Conv_output_0, %/model.6/m.0/cv2/act/Sigmoid_output_0)
#   %/model.6/m.0/Add_output_0 = Add(%/model.6/Slice_1_output_0, %/model.6/m.0/cv2/act/Mul_output_0)
#   %/model.6/m.1/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.6/m.0/Add_output_0, %onnx::Conv_1004, %onnx::Conv_1005)
#   %/model.6/m.1/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.6/m.1/cv1/conv/Conv_output_0)
#   %/model.6/m.1/cv1/act/Mul_output_0 = Mul(%/model.6/m.1/cv1/conv/Conv_output_0, %/model.6/m.1/cv1/act/Sigmoid_output_0)
#   %/model.6/m.1/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.6/m.1/cv1/act/Mul_output_0, %onnx::Conv_1007, %onnx::Conv_1008)
#   %/model.6/m.1/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.6/m.1/cv2/conv/Conv_output_0)
#   %/model.6/m.1/cv2/act/Mul_output_0 = Mul(%/model.6/m.1/cv2/conv/Conv_output_0, %/model.6/m.1/cv2/act/Sigmoid_output_0)
#   %/model.6/m.1/Add_output_0 = Add(%/model.6/m.0/Add_output_0, %/model.6/m.1/cv2/act/Mul_output_0)
#   %/model.6/Concat_output_0 = Concat[axis = 1](%/model.6/Slice_output_0, %/model.6/Slice_1_output_0, %/model.6/m.0/Add_output_0, %/model.6/m.1/Add_output_0)
#   %/model.6/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.6/Concat_output_0, %onnx::Conv_1010, %onnx::Conv_1011)
#   %/model.6/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.6/cv2/conv/Conv_output_0)
#   %/model.6/cv2/act/Mul_output_0 = Mul(%/model.6/cv2/conv/Conv_output_0, %/model.6/cv2/act/Sigmoid_output_0)
#   %/model.7/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]](%/model.6/cv2/act/Mul_output_0, %onnx::Conv_1013, %onnx::Conv_1014)
#   %/model.7/act/Sigmoid_output_0 = Sigmoid(%/model.7/conv/Conv_output_0)
#   %/model.7/act/Mul_output_0 = Mul(%/model.7/conv/Conv_output_0, %/model.7/act/Sigmoid_output_0)
#   %/model.8/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.7/act/Mul_output_0, %onnx::Conv_1016, %onnx::Conv_1017)
#   %/model.8/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.8/cv1/conv/Conv_output_0)
#   %/model.8/cv1/act/Mul_output_0 = Mul(%/model.8/cv1/conv/Conv_output_0, %/model.8/cv1/act/Sigmoid_output_0)
#   %/model.8/Shape_output_0 = Shape(%/model.8/cv1/act/Mul_output_0)
#   %/model.8/Constant_output_0 = Constant[value = <Tensor>]()
#   %/model.8/Gather_output_0 = Gather[axis = 0](%/model.8/Shape_output_0, %/model.8/Constant_output_0)
#   %/model.8/Constant_1_output_0 = Constant[value = <Tensor>]()
#   %/model.8/Constant_2_output_0 = Constant[value = <Tensor>]()
#   %/model.8/Add_output_0 = Add(%/model.8/Gather_output_0, %/model.8/Constant_2_output_0)
#   %/model.8/Constant_3_output_0 = Constant[value = <Tensor>]()
#   %/model.8/Div_output_0 = Div(%/model.8/Add_output_0, %/model.8/Constant_3_output_0)
#   %/model.8/Constant_4_output_0 = Constant[value = <Tensor>]()
#   %/model.8/Mul_output_0 = Mul(%/model.8/Div_output_0, %/model.8/Constant_4_output_0)
#   %/model.8/Slice_output_0 = Slice(%/model.8/cv1/act/Mul_output_0, %/model.8/Constant_1_output_0, %/model.8/Mul_output_0, %/model.8/Constant_output_0)
#   %/model.8/Constant_5_output_0 = Constant[value = <Tensor>]()
#   %/model.8/Mul_1_output_0 = Mul(%/model.8/Div_output_0, %/model.8/Constant_5_output_0)
#   %/model.8/Slice_1_output_0 = Slice(%/model.8/cv1/act/Mul_output_0, %/model.8/Mul_output_0, %/model.8/Mul_1_output_0, %/model.8/Constant_output_0)
#   %/model.8/m.0/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.8/Slice_1_output_0, %onnx::Conv_1019, %onnx::Conv_1020)
#   %/model.8/m.0/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.8/m.0/cv1/conv/Conv_output_0)
#   %/model.8/m.0/cv1/act/Mul_output_0 = Mul(%/model.8/m.0/cv1/conv/Conv_output_0, %/model.8/m.0/cv1/act/Sigmoid_output_0)
#   %/model.8/m.0/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.8/m.0/cv1/act/Mul_output_0, %onnx::Conv_1022, %onnx::Conv_1023)
#   %/model.8/m.0/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.8/m.0/cv2/conv/Conv_output_0)
#   %/model.8/m.0/cv2/act/Mul_output_0 = Mul(%/model.8/m.0/cv2/conv/Conv_output_0, %/model.8/m.0/cv2/act/Sigmoid_output_0)
#   %/model.8/m.0/Add_output_0 = Add(%/model.8/Slice_1_output_0, %/model.8/m.0/cv2/act/Mul_output_0)
#   %/model.8/Concat_output_0 = Concat[axis = 1](%/model.8/Slice_output_0, %/model.8/Slice_1_output_0, %/model.8/m.0/Add_output_0)
#   %/model.8/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.8/Concat_output_0, %onnx::Conv_1025, %onnx::Conv_1026)
#   %/model.8/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.8/cv2/conv/Conv_output_0)
#   %/model.8/cv2/act/Mul_output_0 = Mul(%/model.8/cv2/conv/Conv_output_0, %/model.8/cv2/act/Sigmoid_output_0)
#   %/model.9/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.8/cv2/act/Mul_output_0, %onnx::Conv_1028, %onnx::Conv_1029)
#   %/model.9/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.9/cv1/conv/Conv_output_0)
#   %/model.9/cv1/act/Mul_output_0 = Mul(%/model.9/cv1/conv/Conv_output_0, %/model.9/cv1/act/Sigmoid_output_0)
#   %/model.9/m/MaxPool_output_0 = MaxPool[ceil_mode = 0, dilations = [1, 1], kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]](%/model.9/cv1/act/Mul_output_0)
#   %/model.9/m_1/MaxPool_output_0 = MaxPool[ceil_mode = 0, dilations = [1, 1], kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]](%/model.9/m/MaxPool_output_0)
#   %/model.9/m_2/MaxPool_output_0 = MaxPool[ceil_mode = 0, dilations = [1, 1], kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]](%/model.9/m_1/MaxPool_output_0)
#   %/model.9/Concat_output_0 = Concat[axis = 1](%/model.9/cv1/act/Mul_output_0, %/model.9/m/MaxPool_output_0, %/model.9/m_1/MaxPool_output_0, %/model.9/m_2/MaxPool_output_0)
#   %/model.9/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.9/Concat_output_0, %onnx::Conv_1031, %onnx::Conv_1032)
#   %/model.9/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.9/cv2/conv/Conv_output_0)
#   %/model.9/cv2/act/Mul_output_0 = Mul(%/model.9/cv2/conv/Conv_output_0, %/model.9/cv2/act/Sigmoid_output_0)
#   %/model.10/Constant_output_0 = Constant[value = <Tensor>]()
#   %/model.10/Constant_1_output_0 = Constant[value = <Tensor>]()
#   %/model.10/Resize_output_0 = Resize[coordinate_transformation_mode = 'asymmetric', cubic_coeff_a = -0.75, mode = 'nearest', nearest_mode = 'floor'](%/model.9/cv2/act/Mul_output_0, %/model.10/Constant_1_output_0, %/model.10/Constant_output_0)
#   %/model.11/Concat_output_0 = Concat[axis = 1](%/model.10/Resize_output_0, %/model.6/cv2/act/Mul_output_0)
#   %/model.12/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.11/Concat_output_0, %onnx::Conv_1034, %onnx::Conv_1035)
#   %/model.12/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.12/cv1/conv/Conv_output_0)
#   %/model.12/cv1/act/Mul_output_0 = Mul(%/model.12/cv1/conv/Conv_output_0, %/model.12/cv1/act/Sigmoid_output_0)
#   %/model.12/Shape_output_0 = Shape(%/model.12/cv1/act/Mul_output_0)
#   %/model.12/Constant_output_0 = Constant[value = <Tensor>]()
#   %/model.12/Gather_output_0 = Gather[axis = 0](%/model.12/Shape_output_0, %/model.12/Constant_output_0)
#   %/model.12/Constant_1_output_0 = Constant[value = <Tensor>]()
#   %/model.12/Constant_2_output_0 = Constant[value = <Tensor>]()
#   %/model.12/Add_output_0 = Add(%/model.12/Gather_output_0, %/model.12/Constant_2_output_0)
#   %/model.12/Constant_3_output_0 = Constant[value = <Tensor>]()
#   %/model.12/Div_output_0 = Div(%/model.12/Add_output_0, %/model.12/Constant_3_output_0)
#   %/model.12/Constant_4_output_0 = Constant[value = <Tensor>]()
#   %/model.12/Mul_output_0 = Mul(%/model.12/Div_output_0, %/model.12/Constant_4_output_0)
#   %/model.12/Slice_output_0 = Slice(%/model.12/cv1/act/Mul_output_0, %/model.12/Constant_1_output_0, %/model.12/Mul_output_0, %/model.12/Constant_output_0)
#   %/model.12/Constant_5_output_0 = Constant[value = <Tensor>]()
#   %/model.12/Mul_1_output_0 = Mul(%/model.12/Div_output_0, %/model.12/Constant_5_output_0)
#   %/model.12/Slice_1_output_0 = Slice(%/model.12/cv1/act/Mul_output_0, %/model.12/Mul_output_0, %/model.12/Mul_1_output_0, %/model.12/Constant_output_0)
#   %/model.12/m.0/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.12/Slice_1_output_0, %onnx::Conv_1037, %onnx::Conv_1038)
#   %/model.12/m.0/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.12/m.0/cv1/conv/Conv_output_0)
#   %/model.12/m.0/cv1/act/Mul_output_0 = Mul(%/model.12/m.0/cv1/conv/Conv_output_0, %/model.12/m.0/cv1/act/Sigmoid_output_0)
#   %/model.12/m.0/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.12/m.0/cv1/act/Mul_output_0, %onnx::Conv_1040, %onnx::Conv_1041)
#   %/model.12/m.0/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.12/m.0/cv2/conv/Conv_output_0)
#   %/model.12/m.0/cv2/act/Mul_output_0 = Mul(%/model.12/m.0/cv2/conv/Conv_output_0, %/model.12/m.0/cv2/act/Sigmoid_output_0)
#   %/model.12/Concat_output_0 = Concat[axis = 1](%/model.12/Slice_output_0, %/model.12/Slice_1_output_0, %/model.12/m.0/cv2/act/Mul_output_0)      
#   %/model.12/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.12/Concat_output_0, %onnx::Conv_1043, %onnx::Conv_1044)
#   %/model.12/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.12/cv2/conv/Conv_output_0)
#   %/model.12/cv2/act/Mul_output_0 = Mul(%/model.12/cv2/conv/Conv_output_0, %/model.12/cv2/act/Sigmoid_output_0)
#   %/model.13/Constant_output_0 = Constant[value = <Tensor>]()
#   %/model.13/Constant_1_output_0 = Constant[value = <Tensor>]()
#   %/model.13/Resize_output_0 = Resize[coordinate_transformation_mode = 'asymmetric', cubic_coeff_a = -0.75, mode = 'nearest', nearest_mode = 'floor'](%/model.12/cv2/act/Mul_output_0, %/model.13/Constant_1_output_0, %/model.13/Constant_output_0)
#   %/model.14/Concat_output_0 = Concat[axis = 1](%/model.13/Resize_output_0, %/model.4/cv2/act/Mul_output_0)
#   %/model.15/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.14/Concat_output_0, %onnx::Conv_1046, %onnx::Conv_1047)
#   %/model.15/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.15/cv1/conv/Conv_output_0)
#   %/model.15/cv1/act/Mul_output_0 = Mul(%/model.15/cv1/conv/Conv_output_0, %/model.15/cv1/act/Sigmoid_output_0)
#   %/model.15/Shape_output_0 = Shape(%/model.15/cv1/act/Mul_output_0)
#   %/model.15/Constant_output_0 = Constant[value = <Tensor>]()
#   %/model.15/Gather_output_0 = Gather[axis = 0](%/model.15/Shape_output_0, %/model.15/Constant_output_0)
#   %/model.15/Constant_1_output_0 = Constant[value = <Tensor>]()
#   %/model.15/Constant_2_output_0 = Constant[value = <Tensor>]()
#   %/model.15/Add_output_0 = Add(%/model.15/Gather_output_0, %/model.15/Constant_2_output_0)
#   %/model.15/Constant_3_output_0 = Constant[value = <Tensor>]()
#   %/model.15/Div_output_0 = Div(%/model.15/Add_output_0, %/model.15/Constant_3_output_0)
#   %/model.15/Constant_4_output_0 = Constant[value = <Tensor>]()
#   %/model.15/Mul_output_0 = Mul(%/model.15/Div_output_0, %/model.15/Constant_4_output_0)
#   %/model.15/Slice_output_0 = Slice(%/model.15/cv1/act/Mul_output_0, %/model.15/Constant_1_output_0, %/model.15/Mul_output_0, %/model.15/Constant_output_0)
#   %/model.15/Constant_5_output_0 = Constant[value = <Tensor>]()
#   %/model.15/Mul_1_output_0 = Mul(%/model.15/Div_output_0, %/model.15/Constant_5_output_0)
#   %/model.15/Slice_1_output_0 = Slice(%/model.15/cv1/act/Mul_output_0, %/model.15/Mul_output_0, %/model.15/Mul_1_output_0, %/model.15/Constant_output_0)
#   %/model.15/m.0/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.15/Slice_1_output_0, %onnx::Conv_1049, %onnx::Conv_1050)
#   %/model.15/m.0/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.15/m.0/cv1/conv/Conv_output_0)
#   %/model.15/m.0/cv1/act/Mul_output_0 = Mul(%/model.15/m.0/cv1/conv/Conv_output_0, %/model.15/m.0/cv1/act/Sigmoid_output_0)
#   %/model.15/m.0/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.15/m.0/cv1/act/Mul_output_0, %onnx::Conv_1052, %onnx::Conv_1053)
#   %/model.15/m.0/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.15/m.0/cv2/conv/Conv_output_0)
#   %/model.15/m.0/cv2/act/Mul_output_0 = Mul(%/model.15/m.0/cv2/conv/Conv_output_0, %/model.15/m.0/cv2/act/Sigmoid_output_0)
#   %/model.15/Concat_output_0 = Concat[axis = 1](%/model.15/Slice_output_0, %/model.15/Slice_1_output_0, %/model.15/m.0/cv2/act/Mul_output_0)      
#   %/model.15/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.15/Concat_output_0, %onnx::Conv_1055, %onnx::Conv_1056)
#   %/model.15/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.15/cv2/conv/Conv_output_0)
#   %/model.15/cv2/act/Mul_output_0 = Mul(%/model.15/cv2/conv/Conv_output_0, %/model.15/cv2/act/Sigmoid_output_0)
#   %/model.16/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]](%/model.15/cv2/act/Mul_output_0, %onnx::Conv_1058, %onnx::Conv_1059)
#   %/model.16/act/Sigmoid_output_0 = Sigmoid(%/model.16/conv/Conv_output_0)
#   %/model.16/act/Mul_output_0 = Mul(%/model.16/conv/Conv_output_0, %/model.16/act/Sigmoid_output_0)
#   %/model.17/Concat_output_0 = Concat[axis = 1](%/model.16/act/Mul_output_0, %/model.12/cv2/act/Mul_output_0)
#   %/model.18/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.17/Concat_output_0, %onnx::Conv_1061, %onnx::Conv_1062)
#   %/model.18/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.18/cv1/conv/Conv_output_0)
#   %/model.18/cv1/act/Mul_output_0 = Mul(%/model.18/cv1/conv/Conv_output_0, %/model.18/cv1/act/Sigmoid_output_0)
#   %/model.18/Shape_output_0 = Shape(%/model.18/cv1/act/Mul_output_0)
#   %/model.18/Constant_output_0 = Constant[value = <Tensor>]()
#   %/model.18/Gather_output_0 = Gather[axis = 0](%/model.18/Shape_output_0, %/model.18/Constant_output_0)
#   %/model.18/Constant_1_output_0 = Constant[value = <Tensor>]()
#   %/model.18/Constant_2_output_0 = Constant[value = <Tensor>]()
#   %/model.18/Add_output_0 = Add(%/model.18/Gather_output_0, %/model.18/Constant_2_output_0)
#   %/model.18/Constant_3_output_0 = Constant[value = <Tensor>]()
#   %/model.18/Div_output_0 = Div(%/model.18/Add_output_0, %/model.18/Constant_3_output_0)
#   %/model.18/Constant_4_output_0 = Constant[value = <Tensor>]()
#   %/model.18/Mul_output_0 = Mul(%/model.18/Div_output_0, %/model.18/Constant_4_output_0)
#   %/model.18/Slice_output_0 = Slice(%/model.18/cv1/act/Mul_output_0, %/model.18/Constant_1_output_0, %/model.18/Mul_output_0, %/model.18/Constant_output_0)
#   %/model.18/Constant_5_output_0 = Constant[value = <Tensor>]()
#   %/model.18/Mul_1_output_0 = Mul(%/model.18/Div_output_0, %/model.18/Constant_5_output_0)
#   %/model.18/Slice_1_output_0 = Slice(%/model.18/cv1/act/Mul_output_0, %/model.18/Mul_output_0, %/model.18/Mul_1_output_0, %/model.18/Constant_output_0)
#   %/model.18/m.0/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.18/Slice_1_output_0, %onnx::Conv_1064, %onnx::Conv_1065)
#   %/model.18/m.0/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.18/m.0/cv1/conv/Conv_output_0)
#   %/model.18/m.0/cv1/act/Mul_output_0 = Mul(%/model.18/m.0/cv1/conv/Conv_output_0, %/model.18/m.0/cv1/act/Sigmoid_output_0)
#   %/model.18/m.0/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.18/m.0/cv1/act/Mul_output_0, %onnx::Conv_1067, %onnx::Conv_1068)
#   %/model.18/m.0/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.18/m.0/cv2/conv/Conv_output_0)
#   %/model.18/m.0/cv2/act/Mul_output_0 = Mul(%/model.18/m.0/cv2/conv/Conv_output_0, %/model.18/m.0/cv2/act/Sigmoid_output_0)
#   %/model.18/Concat_output_0 = Concat[axis = 1](%/model.18/Slice_output_0, %/model.18/Slice_1_output_0, %/model.18/m.0/cv2/act/Mul_output_0)      
#   %/model.18/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.18/Concat_output_0, %onnx::Conv_1070, %onnx::Conv_1071)
#   %/model.18/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.18/cv2/conv/Conv_output_0)
#   %/model.18/cv2/act/Mul_output_0 = Mul(%/model.18/cv2/conv/Conv_output_0, %/model.18/cv2/act/Sigmoid_output_0)
#   %/model.19/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]](%/model.18/cv2/act/Mul_output_0, %onnx::Conv_1073, %onnx::Conv_1074)
#   %/model.19/act/Sigmoid_output_0 = Sigmoid(%/model.19/conv/Conv_output_0)
#   %/model.19/act/Mul_output_0 = Mul(%/model.19/conv/Conv_output_0, %/model.19/act/Sigmoid_output_0)
#   %/model.20/Concat_output_0 = Concat[axis = 1](%/model.19/act/Mul_output_0, %/model.9/cv2/act/Mul_output_0)
#   %/model.21/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.20/Concat_output_0, %onnx::Conv_1076, %onnx::Conv_1077)
#   %/model.21/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.21/cv1/conv/Conv_output_0)
#   %/model.21/cv1/act/Mul_output_0 = Mul(%/model.21/cv1/conv/Conv_output_0, %/model.21/cv1/act/Sigmoid_output_0)
#   %/model.21/Shape_output_0 = Shape(%/model.21/cv1/act/Mul_output_0)
#   %/model.21/Constant_output_0 = Constant[value = <Tensor>]()
#   %/model.21/Gather_output_0 = Gather[axis = 0](%/model.21/Shape_output_0, %/model.21/Constant_output_0)
#   %/model.21/Constant_1_output_0 = Constant[value = <Tensor>]()
#   %/model.21/Constant_2_output_0 = Constant[value = <Tensor>]()
#   %/model.21/Add_output_0 = Add(%/model.21/Gather_output_0, %/model.21/Constant_2_output_0)
#   %/model.21/Constant_3_output_0 = Constant[value = <Tensor>]()
#   %/model.21/Div_output_0 = Div(%/model.21/Add_output_0, %/model.21/Constant_3_output_0)
#   %/model.21/Constant_4_output_0 = Constant[value = <Tensor>]()
#   %/model.21/Mul_output_0 = Mul(%/model.21/Div_output_0, %/model.21/Constant_4_output_0)
#   %/model.21/Slice_output_0 = Slice(%/model.21/cv1/act/Mul_output_0, %/model.21/Constant_1_output_0, %/model.21/Mul_output_0, %/model.21/Constant_output_0)
#   %/model.21/Constant_5_output_0 = Constant[value = <Tensor>]()
#   %/model.21/Mul_1_output_0 = Mul(%/model.21/Div_output_0, %/model.21/Constant_5_output_0)
#   %/model.21/Slice_1_output_0 = Slice(%/model.21/cv1/act/Mul_output_0, %/model.21/Mul_output_0, %/model.21/Mul_1_output_0, %/model.21/Constant_output_0)
#   %/model.21/m.0/cv1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.21/Slice_1_output_0, %onnx::Conv_1079, %onnx::Conv_1080)
#   %/model.21/m.0/cv1/act/Sigmoid_output_0 = Sigmoid(%/model.21/m.0/cv1/conv/Conv_output_0)
#   %/model.21/m.0/cv1/act/Mul_output_0 = Mul(%/model.21/m.0/cv1/conv/Conv_output_0, %/model.21/m.0/cv1/act/Sigmoid_output_0)
#   %/model.21/m.0/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.21/m.0/cv1/act/Mul_output_0, %onnx::Conv_1082, %onnx::Conv_1083)
#   %/model.21/m.0/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.21/m.0/cv2/conv/Conv_output_0)
#   %/model.21/m.0/cv2/act/Mul_output_0 = Mul(%/model.21/m.0/cv2/conv/Conv_output_0, %/model.21/m.0/cv2/act/Sigmoid_output_0)
#   %/model.21/Concat_output_0 = Concat[axis = 1](%/model.21/Slice_output_0, %/model.21/Slice_1_output_0, %/model.21/m.0/cv2/act/Mul_output_0)      
#   %/model.21/cv2/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.21/Concat_output_0, %onnx::Conv_1085, %onnx::Conv_1086)
#   %/model.21/cv2/act/Sigmoid_output_0 = Sigmoid(%/model.21/cv2/conv/Conv_output_0)
#   %/model.21/cv2/act/Mul_output_0 = Mul(%/model.21/cv2/conv/Conv_output_0, %/model.21/cv2/act/Sigmoid_output_0)
#   %/model.22/cv2.0/cv2.0.0/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.15/cv2/act/Mul_output_0, %onnx::Conv_1088, %onnx::Conv_1089)
#   %/model.22/cv2.0/cv2.0.0/act/Sigmoid_output_0 = Sigmoid(%/model.22/cv2.0/cv2.0.0/conv/Conv_output_0)
#   %/model.22/cv2.0/cv2.0.0/act/Mul_output_0 = Mul(%/model.22/cv2.0/cv2.0.0/conv/Conv_output_0, %/model.22/cv2.0/cv2.0.0/act/Sigmoid_output_0)     
#   %/model.22/cv2.0/cv2.0.1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.22/cv2.0/cv2.0.0/act/Mul_output_0, %onnx::Conv_1091, %onnx::Conv_1092)
#   %/model.22/cv2.0/cv2.0.1/act/Sigmoid_output_0 = Sigmoid(%/model.22/cv2.0/cv2.0.1/conv/Conv_output_0)
#   %/model.22/cv2.0/cv2.0.1/act/Mul_output_0 = Mul(%/model.22/cv2.0/cv2.0.1/conv/Conv_output_0, %/model.22/cv2.0/cv2.0.1/act/Sigmoid_output_0)     
#   %/model.22/cv2.0/cv2.0.2/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.22/cv2.0/cv2.0.1/act/Mul_output_0, %model.22.cv2.0.2.weight, %model.22.cv2.0.2.bias)
#   %/model.22/cv3.0/cv3.0.0/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.15/cv2/act/Mul_output_0, %onnx::Conv_1094, %onnx::Conv_1095)
#   %/model.22/cv3.0/cv3.0.0/act/Sigmoid_output_0 = Sigmoid(%/model.22/cv3.0/cv3.0.0/conv/Conv_output_0)
#   %/model.22/cv3.0/cv3.0.0/act/Mul_output_0 = Mul(%/model.22/cv3.0/cv3.0.0/conv/Conv_output_0, %/model.22/cv3.0/cv3.0.0/act/Sigmoid_output_0)     
#   %/model.22/cv3.0/cv3.0.1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.22/cv3.0/cv3.0.0/act/Mul_output_0, %onnx::Conv_1097, %onnx::Conv_1098)
#   %/model.22/cv3.0/cv3.0.1/act/Sigmoid_output_0 = Sigmoid(%/model.22/cv3.0/cv3.0.1/conv/Conv_output_0)
#   %/model.22/cv3.0/cv3.0.1/act/Mul_output_0 = Mul(%/model.22/cv3.0/cv3.0.1/conv/Conv_output_0, %/model.22/cv3.0/cv3.0.1/act/Sigmoid_output_0)     
#   %/model.22/cv3.0/cv3.0.2/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.22/cv3.0/cv3.0.1/act/Mul_output_0, %model.22.cv3.0.2.weight, %model.22.cv3.0.2.bias)
#   %onnx::Shape_699 = Concat[axis = 1](%/model.22/cv2.0/cv2.0.2/Conv_output_0, %/model.22/cv3.0/cv3.0.2/Conv_output_0)
#   %/model.22/cv2.1/cv2.1.0/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.18/cv2/act/Mul_output_0, %onnx::Conv_1100, %onnx::Conv_1101)
#   %/model.22/cv2.1/cv2.1.0/act/Sigmoid_output_0 = Sigmoid(%/model.22/cv2.1/cv2.1.0/conv/Conv_output_0)
#   %/model.22/cv2.1/cv2.1.0/act/Mul_output_0 = Mul(%/model.22/cv2.1/cv2.1.0/conv/Conv_output_0, %/model.22/cv2.1/cv2.1.0/act/Sigmoid_output_0)     
#   %/model.22/cv2.1/cv2.1.1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.22/cv2.1/cv2.1.0/act/Mul_output_0, %onnx::Conv_1103, %onnx::Conv_1104)
#   %/model.22/cv2.1/cv2.1.1/act/Sigmoid_output_0 = Sigmoid(%/model.22/cv2.1/cv2.1.1/conv/Conv_output_0)
#   %/model.22/cv2.1/cv2.1.1/act/Mul_output_0 = Mul(%/model.22/cv2.1/cv2.1.1/conv/Conv_output_0, %/model.22/cv2.1/cv2.1.1/act/Sigmoid_output_0)     
#   %/model.22/cv2.1/cv2.1.2/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.22/cv2.1/cv2.1.1/act/Mul_output_0, %model.22.cv2.1.2.weight, %model.22.cv2.1.2.bias)
#   %/model.22/cv3.1/cv3.1.0/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.18/cv2/act/Mul_output_0, %onnx::Conv_1106, %onnx::Conv_1107)
#   %/model.22/cv3.1/cv3.1.0/act/Sigmoid_output_0 = Sigmoid(%/model.22/cv3.1/cv3.1.0/conv/Conv_output_0)
#   %/model.22/cv3.1/cv3.1.0/act/Mul_output_0 = Mul(%/model.22/cv3.1/cv3.1.0/conv/Conv_output_0, %/model.22/cv3.1/cv3.1.0/act/Sigmoid_output_0)     
#   %/model.22/cv3.1/cv3.1.1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.22/cv3.1/cv3.1.0/act/Mul_output_0, %onnx::Conv_1109, %onnx::Conv_1110)
#   %/model.22/cv3.1/cv3.1.1/act/Sigmoid_output_0 = Sigmoid(%/model.22/cv3.1/cv3.1.1/conv/Conv_output_0)
#   %/model.22/cv3.1/cv3.1.1/act/Mul_output_0 = Mul(%/model.22/cv3.1/cv3.1.1/conv/Conv_output_0, %/model.22/cv3.1/cv3.1.1/act/Sigmoid_output_0)     
#   %/model.22/cv3.1/cv3.1.2/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.22/cv3.1/cv3.1.1/act/Mul_output_0, %model.22.cv3.1.2.weight, %model.22.cv3.1.2.bias)
#   %onnx::Reshape_718 = Concat[axis = 1](%/model.22/cv2.1/cv2.1.2/Conv_output_0, %/model.22/cv3.1/cv3.1.2/Conv_output_0)
#   %/model.22/cv2.2/cv2.2.0/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.21/cv2/act/Mul_output_0, %onnx::Conv_1112, %onnx::Conv_1113)
#   %/model.22/cv2.2/cv2.2.0/act/Sigmoid_output_0 = Sigmoid(%/model.22/cv2.2/cv2.2.0/conv/Conv_output_0)
#   %/model.22/cv2.2/cv2.2.0/act/Mul_output_0 = Mul(%/model.22/cv2.2/cv2.2.0/conv/Conv_output_0, %/model.22/cv2.2/cv2.2.0/act/Sigmoid_output_0)     
#   %/model.22/cv2.2/cv2.2.1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.22/cv2.2/cv2.2.0/act/Mul_output_0, %onnx::Conv_1115, %onnx::Conv_1116)
#   %/model.22/cv2.2/cv2.2.1/act/Sigmoid_output_0 = Sigmoid(%/model.22/cv2.2/cv2.2.1/conv/Conv_output_0)
#   %/model.22/cv2.2/cv2.2.1/act/Mul_output_0 = Mul(%/model.22/cv2.2/cv2.2.1/conv/Conv_output_0, %/model.22/cv2.2/cv2.2.1/act/Sigmoid_output_0)     
#   %/model.22/cv2.2/cv2.2.2/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.22/cv2.2/cv2.2.1/act/Mul_output_0, %model.22.cv2.2.2.weight, %model.22.cv2.2.2.bias)
#   %/model.22/cv3.2/cv3.2.0/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.21/cv2/act/Mul_output_0, %onnx::Conv_1118, %onnx::Conv_1119)
#   %/model.22/cv3.2/cv3.2.0/act/Sigmoid_output_0 = Sigmoid(%/model.22/cv3.2/cv3.2.0/conv/Conv_output_0)
#   %/model.22/cv3.2/cv3.2.0/act/Mul_output_0 = Mul(%/model.22/cv3.2/cv3.2.0/conv/Conv_output_0, %/model.22/cv3.2/cv3.2.0/act/Sigmoid_output_0)     
#   %/model.22/cv3.2/cv3.2.1/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/model.22/cv3.2/cv3.2.0/act/Mul_output_0, %onnx::Conv_1121, %onnx::Conv_1122)
#   %/model.22/cv3.2/cv3.2.1/act/Sigmoid_output_0 = Sigmoid(%/model.22/cv3.2/cv3.2.1/conv/Conv_output_0)
#   %/model.22/cv3.2/cv3.2.1/act/Mul_output_0 = Mul(%/model.22/cv3.2/cv3.2.1/conv/Conv_output_0, %/model.22/cv3.2/cv3.2.1/act/Sigmoid_output_0)     
#   %/model.22/cv3.2/cv3.2.2/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.22/cv3.2/cv3.2.1/act/Mul_output_0, %model.22.cv3.2.2.weight, %model.22.cv3.2.2.bias)
#   %onnx::Reshape_737 = Concat[axis = 1](%/model.22/cv2.2/cv2.2.2/Conv_output_0, %/model.22/cv3.2/cv3.2.2/Conv_output_0)
#   %/model.22/Shape_output_0 = Shape(%onnx::Shape_699)
#   %/model.22/Constant_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Gather_output_0 = Gather[axis = 0](%/model.22/Shape_output_0, %/model.22/Constant_output_0)
#   %/model.22/Unsqueeze_output_0 = Unsqueeze[axes = [0]](%/model.22/Gather_output_0)
#   %/model.22/Constant_1_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Constant_2_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Concat_3_output_0 = Concat[axis = 0](%/model.22/Unsqueeze_output_0, %/model.22/Constant_1_output_0, %/model.22/Constant_2_output_0)  
#   %/model.22/Unsqueeze_1_output_0 = Unsqueeze[axes = [0]](%/model.22/Gather_output_0)
#   %/model.22/Constant_3_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Constant_4_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Concat_4_output_0 = Concat[axis = 0](%/model.22/Unsqueeze_1_output_0, %/model.22/Constant_3_output_0, %/model.22/Constant_4_output_0)  %/model.22/Unsqueeze_2_output_0 = Unsqueeze[axes = [0]](%/model.22/Gather_output_0)
#   %/model.22/Constant_5_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Constant_6_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Concat_5_output_0 = Concat[axis = 0](%/model.22/Unsqueeze_2_output_0, %/model.22/Constant_5_output_0, %/model.22/Constant_6_output_0)  %/model.22/Reshape_output_0 = Reshape(%onnx::Shape_699, %/model.22/Concat_3_output_0)
#   %/model.22/Reshape_1_output_0 = Reshape(%onnx::Reshape_718, %/model.22/Concat_4_output_0)
#   %/model.22/Reshape_2_output_0 = Reshape(%onnx::Reshape_737, %/model.22/Concat_5_output_0)
#   %/model.22/Concat_6_output_0 = Concat[axis = 2](%/model.22/Reshape_output_0, %/model.22/Reshape_1_output_0, %/model.22/Reshape_2_output_0)      
#   %/model.22/Constant_7_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Split_output_0, %/model.22/Split_output_1, %/model.22/Split_output_2 = Split[axis = 0, split = [1, 1, 1]](%/model.22/Constant_7_output_0)
#   %/model.22/Squeeze_output_0 = Squeeze[axes = [0]](%/model.22/Split_output_0)
#   %/model.22/Squeeze_1_output_0 = Squeeze[axes = [0]](%/model.22/Split_output_1)
#   %/model.22/Squeeze_2_output_0 = Squeeze[axes = [0]](%/model.22/Split_output_2)
#   %/model.22/Shape_1_output_0 = Shape(%onnx::Shape_699)
#   %/model.22/Constant_8_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Gather_1_output_0 = Gather[axis = 0](%/model.22/Shape_1_output_0, %/model.22/Constant_8_output_0)
#   %/model.22/Shape_2_output_0 = Shape(%onnx::Shape_699)
#   %/model.22/Constant_9_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Gather_2_output_0 = Gather[axis = 0](%/model.22/Shape_2_output_0, %/model.22/Constant_9_output_0)
#   %/model.22/Cast_output_0 = Cast[to = 1](%/model.22/Gather_2_output_0)
#   %/model.22/Constant_10_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Constant_11_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Range_output_0 = Range(%/model.22/Constant_10_output_0, %/model.22/Cast_output_0, %/model.22/Constant_11_output_0)
#   %/model.22/Constant_12_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Add_output_0 = Add(%/model.22/Range_output_0, %/model.22/Constant_12_output_0)
#   %/model.22/Cast_1_output_0 = Cast[to = 1](%/model.22/Gather_1_output_0)
#   %/model.22/Constant_13_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Constant_14_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Range_1_output_0 = Range(%/model.22/Constant_13_output_0, %/model.22/Cast_1_output_0, %/model.22/Constant_14_output_0)
#   %/model.22/Constant_15_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Add_1_output_0 = Add(%/model.22/Range_1_output_0, %/model.22/Constant_15_output_0)
#   %/model.22/Constant_16_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Reshape_3_output_0 = Reshape(%/model.22/Add_1_output_0, %/model.22/Constant_16_output_0)
#   %/model.22/Constant_17_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Reshape_4_output_0 = Reshape(%/model.22/Add_output_0, %/model.22/Constant_17_output_0)
#   %/model.22/Shape_3_output_0 = Shape(%/model.22/Reshape_3_output_0)
#   %/model.22/Shape_4_output_0 = Shape(%/model.22/Reshape_4_output_0)
#   %/model.22/Concat_7_output_0 = Concat[axis = 0](%/model.22/Shape_3_output_0, %/model.22/Shape_4_output_0)
#   %/model.22/Constant_18_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Concat_8_output_0 = Concat[axis = 0](%/model.22/Shape_3_output_0, %/model.22/Constant_18_output_0)
#   %/model.22/Reshape_5_output_0 = Reshape(%/model.22/Reshape_3_output_0, %/model.22/Concat_8_output_0)
#   %/model.22/Expand_output_0 = Expand(%/model.22/Reshape_5_output_0, %/model.22/Concat_7_output_0)
#   %/model.22/Constant_19_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Concat_9_output_0 = Concat[axis = 0](%/model.22/Constant_19_output_0, %/model.22/Shape_4_output_0)
#   %/model.22/Reshape_6_output_0 = Reshape(%/model.22/Reshape_4_output_0, %/model.22/Concat_9_output_0)
#   %/model.22/Expand_1_output_0 = Expand(%/model.22/Reshape_6_output_0, %/model.22/Concat_7_output_0)
#   %/model.22/Unsqueeze_3_output_0 = Unsqueeze[axes = [-1]](%/model.22/Expand_1_output_0)
#   %/model.22/Unsqueeze_4_output_0 = Unsqueeze[axes = [-1]](%/model.22/Expand_output_0)
#   %/model.22/Concat_10_output_0 = Concat[axis = -1](%/model.22/Unsqueeze_3_output_0, %/model.22/Unsqueeze_4_output_0)
#   %/model.22/Constant_20_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Reshape_7_output_0 = Reshape(%/model.22/Concat_10_output_0, %/model.22/Constant_20_output_0)
#   %/model.22/Mul_output_0 = Mul(%/model.22/Gather_1_output_0, %/model.22/Gather_2_output_0)
#   %/model.22/Unsqueeze_5_output_0 = Unsqueeze[axes = [0]](%/model.22/Mul_output_0)
#   %/model.22/Constant_21_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Concat_11_output_0 = Concat[axis = 0](%/model.22/Unsqueeze_5_output_0, %/model.22/Constant_21_output_0)
#   %/model.22/ConstantOfShape_output_0 = ConstantOfShape[value = <Tensor>](%/model.22/Concat_11_output_0)
#   %/model.22/Add_2_output_0 = Add(%/model.22/ConstantOfShape_output_0, %/model.22/Squeeze_output_0)
#   %/model.22/Shape_5_output_0 = Shape(%onnx::Reshape_718)
#   %/model.22/Constant_22_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Gather_3_output_0 = Gather[axis = 0](%/model.22/Shape_5_output_0, %/model.22/Constant_22_output_0)
#   %/model.22/Shape_6_output_0 = Shape(%onnx::Reshape_718)
#   %/model.22/Constant_23_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Gather_4_output_0 = Gather[axis = 0](%/model.22/Shape_6_output_0, %/model.22/Constant_23_output_0)
#   %/model.22/Cast_2_output_0 = Cast[to = 1](%/model.22/Gather_4_output_0)
#   %/model.22/Constant_24_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Constant_25_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Range_2_output_0 = Range(%/model.22/Constant_24_output_0, %/model.22/Cast_2_output_0, %/model.22/Constant_25_output_0)
#   %/model.22/Constant_26_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Add_3_output_0 = Add(%/model.22/Range_2_output_0, %/model.22/Constant_26_output_0)
#   %/model.22/Cast_3_output_0 = Cast[to = 1](%/model.22/Gather_3_output_0)
#   %/model.22/Constant_27_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Constant_28_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Range_3_output_0 = Range(%/model.22/Constant_27_output_0, %/model.22/Cast_3_output_0, %/model.22/Constant_28_output_0)
#   %/model.22/Constant_29_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Add_4_output_0 = Add(%/model.22/Range_3_output_0, %/model.22/Constant_29_output_0)
#   %/model.22/Constant_30_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Reshape_8_output_0 = Reshape(%/model.22/Add_4_output_0, %/model.22/Constant_30_output_0)
#   %/model.22/Constant_31_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Reshape_9_output_0 = Reshape(%/model.22/Add_3_output_0, %/model.22/Constant_31_output_0)
#   %/model.22/Shape_7_output_0 = Shape(%/model.22/Reshape_8_output_0)
#   %/model.22/Shape_8_output_0 = Shape(%/model.22/Reshape_9_output_0)
#   %/model.22/Concat_12_output_0 = Concat[axis = 0](%/model.22/Shape_7_output_0, %/model.22/Shape_8_output_0)
#   %/model.22/Constant_32_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Concat_13_output_0 = Concat[axis = 0](%/model.22/Shape_7_output_0, %/model.22/Constant_32_output_0)
#   %/model.22/Reshape_10_output_0 = Reshape(%/model.22/Reshape_8_output_0, %/model.22/Concat_13_output_0)
#   %/model.22/Expand_2_output_0 = Expand(%/model.22/Reshape_10_output_0, %/model.22/Concat_12_output_0)
#   %/model.22/Constant_33_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Concat_14_output_0 = Concat[axis = 0](%/model.22/Constant_33_output_0, %/model.22/Shape_8_output_0)
#   %/model.22/Reshape_11_output_0 = Reshape(%/model.22/Reshape_9_output_0, %/model.22/Concat_14_output_0)
#   %/model.22/Expand_3_output_0 = Expand(%/model.22/Reshape_11_output_0, %/model.22/Concat_12_output_0)
#   %/model.22/Unsqueeze_6_output_0 = Unsqueeze[axes = [-1]](%/model.22/Expand_3_output_0)
#   %/model.22/Unsqueeze_7_output_0 = Unsqueeze[axes = [-1]](%/model.22/Expand_2_output_0)
#   %/model.22/Concat_15_output_0 = Concat[axis = -1](%/model.22/Unsqueeze_6_output_0, %/model.22/Unsqueeze_7_output_0)
#   %/model.22/Constant_34_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Reshape_12_output_0 = Reshape(%/model.22/Concat_15_output_0, %/model.22/Constant_34_output_0)
#   %/model.22/Mul_1_output_0 = Mul(%/model.22/Gather_3_output_0, %/model.22/Gather_4_output_0)
#   %/model.22/Unsqueeze_8_output_0 = Unsqueeze[axes = [0]](%/model.22/Mul_1_output_0)
#   %/model.22/Constant_35_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Concat_16_output_0 = Concat[axis = 0](%/model.22/Unsqueeze_8_output_0, %/model.22/Constant_35_output_0)
#   %/model.22/ConstantOfShape_1_output_0 = ConstantOfShape[value = <Tensor>](%/model.22/Concat_16_output_0)
#   %/model.22/Add_5_output_0 = Add(%/model.22/ConstantOfShape_1_output_0, %/model.22/Squeeze_1_output_0)
#   %/model.22/Shape_9_output_0 = Shape(%onnx::Reshape_737)
#   %/model.22/Constant_36_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Gather_5_output_0 = Gather[axis = 0](%/model.22/Shape_9_output_0, %/model.22/Constant_36_output_0)
#   %/model.22/Shape_10_output_0 = Shape(%onnx::Reshape_737)
#   %/model.22/Constant_37_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Gather_6_output_0 = Gather[axis = 0](%/model.22/Shape_10_output_0, %/model.22/Constant_37_output_0)
#   %/model.22/Cast_4_output_0 = Cast[to = 1](%/model.22/Gather_6_output_0)
#   %/model.22/Constant_38_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Constant_39_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Range_4_output_0 = Range(%/model.22/Constant_38_output_0, %/model.22/Cast_4_output_0, %/model.22/Constant_39_output_0)
#   %/model.22/Constant_40_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Add_6_output_0 = Add(%/model.22/Range_4_output_0, %/model.22/Constant_40_output_0)
#   %/model.22/Cast_5_output_0 = Cast[to = 1](%/model.22/Gather_5_output_0)
#   %/model.22/Constant_41_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Constant_42_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Range_5_output_0 = Range(%/model.22/Constant_41_output_0, %/model.22/Cast_5_output_0, %/model.22/Constant_42_output_0)
#   %/model.22/Constant_43_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Add_7_output_0 = Add(%/model.22/Range_5_output_0, %/model.22/Constant_43_output_0)
#   %/model.22/Constant_44_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Reshape_13_output_0 = Reshape(%/model.22/Add_7_output_0, %/model.22/Constant_44_output_0)
#   %/model.22/Constant_45_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Reshape_14_output_0 = Reshape(%/model.22/Add_6_output_0, %/model.22/Constant_45_output_0)
#   %/model.22/Shape_11_output_0 = Shape(%/model.22/Reshape_13_output_0)
#   %/model.22/Shape_12_output_0 = Shape(%/model.22/Reshape_14_output_0)
#   %/model.22/Concat_17_output_0 = Concat[axis = 0](%/model.22/Shape_11_output_0, %/model.22/Shape_12_output_0)
#   %/model.22/Constant_46_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Concat_18_output_0 = Concat[axis = 0](%/model.22/Shape_11_output_0, %/model.22/Constant_46_output_0)
#   %/model.22/Reshape_15_output_0 = Reshape(%/model.22/Reshape_13_output_0, %/model.22/Concat_18_output_0)
#   %/model.22/Expand_4_output_0 = Expand(%/model.22/Reshape_15_output_0, %/model.22/Concat_17_output_0)
#   %/model.22/Constant_47_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Concat_19_output_0 = Concat[axis = 0](%/model.22/Constant_47_output_0, %/model.22/Shape_12_output_0)
#   %/model.22/Reshape_16_output_0 = Reshape(%/model.22/Reshape_14_output_0, %/model.22/Concat_19_output_0)
#   %/model.22/Expand_5_output_0 = Expand(%/model.22/Reshape_16_output_0, %/model.22/Concat_17_output_0)
#   %/model.22/Unsqueeze_9_output_0 = Unsqueeze[axes = [-1]](%/model.22/Expand_5_output_0)
#   %/model.22/Unsqueeze_10_output_0 = Unsqueeze[axes = [-1]](%/model.22/Expand_4_output_0)
#   %/model.22/Concat_20_output_0 = Concat[axis = -1](%/model.22/Unsqueeze_9_output_0, %/model.22/Unsqueeze_10_output_0)
#   %/model.22/Constant_48_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Reshape_17_output_0 = Reshape(%/model.22/Concat_20_output_0, %/model.22/Constant_48_output_0)
#   %/model.22/Mul_2_output_0 = Mul(%/model.22/Gather_5_output_0, %/model.22/Gather_6_output_0)
#   %/model.22/Unsqueeze_11_output_0 = Unsqueeze[axes = [0]](%/model.22/Mul_2_output_0)
#   %/model.22/Constant_49_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Concat_21_output_0 = Concat[axis = 0](%/model.22/Unsqueeze_11_output_0, %/model.22/Constant_49_output_0)
#   %/model.22/ConstantOfShape_2_output_0 = ConstantOfShape[value = <Tensor>](%/model.22/Concat_21_output_0)
#   %/model.22/Add_8_output_0 = Add(%/model.22/ConstantOfShape_2_output_0, %/model.22/Squeeze_2_output_0)
#   %/model.22/Concat_22_output_0 = Concat[axis = 0](%/model.22/Reshape_7_output_0, %/model.22/Reshape_12_output_0, %/model.22/Reshape_17_output_0) 
#   %/model.22/Concat_23_output_0 = Concat[axis = 0](%/model.22/Add_2_output_0, %/model.22/Add_5_output_0, %/model.22/Add_8_output_0)
#   %/model.22/Transpose_output_0 = Transpose[perm = [1, 0]](%/model.22/Concat_22_output_0)
#   %/model.22/Transpose_1_output_0 = Transpose[perm = [1, 0]](%/model.22/Concat_23_output_0)
#   %/model.22/Split_1_output_0, %/model.22/Split_1_output_1 = Split[axis = 1, split = [64, 3]](%/model.22/Concat_6_output_0)
#   %/model.22/dfl/Shape_output_0 = Shape(%/model.22/Split_1_output_0)
#   %/model.22/dfl/Constant_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/dfl/Gather_output_0 = Gather[axis = 0](%/model.22/dfl/Shape_output_0, %/model.22/dfl/Constant_output_0)
#   %/model.22/dfl/Shape_1_output_0 = Shape(%/model.22/Split_1_output_0)
#   %/model.22/dfl/Constant_1_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/dfl/Gather_1_output_0 = Gather[axis = 0](%/model.22/dfl/Shape_1_output_0, %/model.22/dfl/Constant_1_output_0)
#   %/model.22/dfl/Unsqueeze_output_0 = Unsqueeze[axes = [0]](%/model.22/dfl/Gather_output_0)
#   %/model.22/dfl/Constant_2_output_0 = Constant[value = <Tensor>]()
#   %/model.22/dfl/Constant_3_output_0 = Constant[value = <Tensor>]()
#   %/model.22/dfl/Unsqueeze_1_output_0 = Unsqueeze[axes = [0]](%/model.22/dfl/Gather_1_output_0)
#   %/model.22/dfl/Concat_output_0 = Concat[axis = 0](%/model.22/dfl/Unsqueeze_output_0, %/model.22/dfl/Constant_2_output_0, %/model.22/dfl/Constant_3_output_0, %/model.22/dfl/Unsqueeze_1_output_0)
#   %/model.22/dfl/Reshape_output_0 = Reshape(%/model.22/Split_1_output_0, %/model.22/dfl/Concat_output_0)
#   %/model.22/dfl/Transpose_output_0 = Transpose[perm = [0, 3, 1, 2]](%/model.22/dfl/Reshape_output_0)
#   %/model.22/dfl/Softmax_output_0 = Softmax[axis = 3](%/model.22/dfl/Transpose_output_0)
#   %/model.22/dfl/Transpose_1_output_0 = Transpose[perm = [0, 3, 2, 1]](%/model.22/dfl/Softmax_output_0)
#   %/model.22/dfl/conv/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]](%/model.22/dfl/Transpose_1_output_0, %model.22.dfl.conv.weight)
#   %/model.22/dfl/Unsqueeze_2_output_0 = Unsqueeze[axes = [0]](%/model.22/dfl/Gather_output_0)
#   %/model.22/dfl/Constant_4_output_0 = Constant[value = <Tensor>]()
#   %/model.22/dfl/Unsqueeze_3_output_0 = Unsqueeze[axes = [0]](%/model.22/dfl/Gather_1_output_0)
#   %/model.22/dfl/Concat_1_output_0 = Concat[axis = 0](%/model.22/dfl/Unsqueeze_2_output_0, %/model.22/dfl/Constant_4_output_0, %/model.22/dfl/Unsqueeze_3_output_0)
#   %/model.22/dfl/Reshape_1_output_0 = Reshape(%/model.22/dfl/conv/Conv_output_0, %/model.22/dfl/Concat_1_output_0)
#   %/model.22/Unsqueeze_12_output_0 = Unsqueeze[axes = [0]](%/model.22/Transpose_output_0)
#   %/model.22/Shape_13_output_0 = Shape(%/model.22/dfl/Reshape_1_output_0)
#   %/model.22/Constant_50_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Gather_7_output_0 = Gather[axis = 0](%/model.22/Shape_13_output_0, %/model.22/Constant_50_output_0)
#   %/model.22/Constant_51_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Constant_52_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Add_9_output_0 = Add(%/model.22/Gather_7_output_0, %/model.22/Constant_52_output_0)
#   %/model.22/Constant_53_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Div_output_0 = Div(%/model.22/Add_9_output_0, %/model.22/Constant_53_output_0)
#   %/model.22/Constant_54_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Mul_3_output_0 = Mul(%/model.22/Div_output_0, %/model.22/Constant_54_output_0)
#   %/model.22/Slice_output_0 = Slice(%/model.22/dfl/Reshape_1_output_0, %/model.22/Constant_51_output_0, %/model.22/Mul_3_output_0, %/model.22/Constant_50_output_0)
#   %/model.22/Constant_55_output_0 = Constant[value = <Tensor>]()
#   %/model.22/Mul_4_output_0 = Mul(%/model.22/Div_output_0, %/model.22/Constant_55_output_0)
#   %/model.22/Slice_1_output_0 = Slice(%/model.22/dfl/Reshape_1_output_0, %/model.22/Mul_3_output_0, %/model.22/Mul_4_output_0, %/model.22/Constant_50_output_0)
#   %/model.22/Sub_output_0 = Sub(%/model.22/Unsqueeze_12_output_0, %/model.22/Slice_output_0)
#   %/model.22/Add_10_output_0 = Add(%/model.22/Unsqueeze_12_output_0, %/model.22/Slice_1_output_0)
#   %/model.22/Add_11_output_0 = Add(%/model.22/Sub_output_0, %/model.22/Add_10_output_0)
#   %/model.22/Constant_56_output_0 = Constant[value = <Scalar Tensor []>]()
#   %/model.22/Div_1_output_0 = Div(%/model.22/Add_11_output_0, %/model.22/Constant_56_output_0)
#   %/model.22/Sub_1_output_0 = Sub(%/model.22/Add_10_output_0, %/model.22/Sub_output_0)
#   %/model.22/Concat_24_output_0 = Concat[axis = 1](%/model.22/Div_1_output_0, %/model.22/Sub_1_output_0)
#   %/model.22/Mul_5_output_0 = Mul(%/model.22/Concat_24_output_0, %/model.22/Transpose_1_output_0)
#   %/model.22/Sigmoid_output_0 = Sigmoid(%/model.22/Split_1_output_1)
#   %output = Concat[axis = 1](%/model.22/Mul_5_output_0, %/model.22/Sigmoid_output_0)
#   return %output, %onnx::Shape_699, %onnx::Reshape_718, %onnx::Reshape_737
# }
