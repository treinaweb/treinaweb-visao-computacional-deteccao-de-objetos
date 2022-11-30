from pixellib.torchbackend.instance import instanceSegmentation

inst = instanceSegmentation()
inst.load_model("model/pointrend_resnet50.pkl")
inst.segmentImage(
  "Img/image4.jpg",
   show_bboxes=True,
  output_image_name="Img/image4_seg.jpg"
)