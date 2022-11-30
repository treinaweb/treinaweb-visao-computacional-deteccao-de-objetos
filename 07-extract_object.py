from pixellib.torchbackend.instance import instanceSegmentation

inst = instanceSegmentation()
inst.load_model("model/pointrend_resnet50.pkl")
inst.segmentImage(
    "Img/image3.jpg",
    show_bboxes=True,
    extract_segmented_objects=True,
    extract_from_box=True,
    save_extracted_objects=True,
    output_image_name="Img/image3_ext.jpg"
)