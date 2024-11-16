from manim import Scene, Rectangle, Text, VGroup, FadeIn, FadeOut, Transform, UP, DOWN, LEFT, RIGHT, Line

class RCNNAnimation(Scene):
    def construct(self):
        # Step 1: Show the input image
        image = Rectangle(width=4, height=4, color="blue")
        image_label = Text("Input Image", font_size=24).next_to(image, direction=UP)
        self.play(FadeIn(image), FadeIn(image_label))
        self.wait(1)

        # Step 2: Convolution Layer 1 (showing filters and feature map)
        conv1_filter = Rectangle(width=1, height=1, color="green", fill_opacity=0.6).shift(LEFT*2 + DOWN)
        conv1_label = Text("Conv1", font_size=24).next_to(conv1_filter, direction=UP)
        feature_map1 = Rectangle(width=3, height=3, color="purple", fill_opacity=0.2).shift(RIGHT*2 + UP*1)

        self.play(FadeIn(conv1_filter), FadeIn(conv1_label))
        self.wait(1)
        self.play(Transform(conv1_filter, feature_map1))
        self.wait(1)

        # Step 3: Activation (e.g., ReLU)
        relu_layer = Rectangle(width=3.2, height=3.2, color="orange", fill_opacity=0.4).shift(RIGHT * 2 + UP * 1)
        relu_label = Text("ReLU Activation", font_size=24).next_to(relu_layer, direction=UP)

        self.play(Transform(feature_map1, relu_layer), FadeIn(relu_label))
        self.wait(1)

        # Step 4: Max Pooling Layer (downsampling the feature map)
        pooling_layer = Rectangle(width=2, height=2, color="yellow", fill_opacity=0.4).shift(RIGHT * 2 + DOWN)
        pooling_label = Text("Max Pooling", font_size=24).next_to(pooling_layer, direction=UP)

        self.play(Transform(relu_layer, pooling_layer), FadeIn(pooling_label))
        self.wait(1)

        # Step 5: Region Proposals
        region_proposals = VGroup(
            Rectangle(width=1.5, height=1, color="red", fill_opacity=0.4).shift(LEFT*2 + DOWN*1),
            Rectangle(width=1.5, height=1, color="green", fill_opacity=0.4).shift(RIGHT*2 + DOWN*1)
        )
        region_label = Text("Region Proposals", font_size=24).next_to(region_proposals, direction=DOWN)
        self.play(FadeIn(region_proposals), FadeIn(region_label))
        self.wait(1)

        # Step 6: Region of Interest (RoI) Pooling (representing pooling on proposals)
        roi_pooling_layer = Rectangle(width=1.8, height=1.8, color="blue", fill_opacity=0.4).shift(RIGHT*3 + DOWN*1.2)
        roi_pooling_label = Text("RoI Pooling", font_size=24).next_to(roi_pooling_layer, direction=UP)

        self.play(Transform(pooling_layer, roi_pooling_layer), FadeIn(roi_pooling_label))
        self.wait(1)

        # Step 7: Final Bounding Box Refinement
        refined_box = Rectangle(width=2, height=1, color="blue", fill_opacity=0.4).shift(RIGHT*3 + DOWN*1.2)
        refined_box_label = Text("Refined Bounding Box", font_size=24).next_to(refined_box, direction=DOWN)

        self.play(Transform(roi_pooling_layer, refined_box), FadeIn(refined_box_label))
        self.wait(1)

        # Step 8: Final Output (Classification Result)
        final_classification = Text("Class: Car", font_size=24).next_to(refined_box, direction=DOWN)
        self.play(FadeIn(final_classification))
        self.wait(1)

        # Step 9: Final Detected Object Output
        final_output = Text("Detected Object", font_size=24).next_to(refined_box, direction=DOWN)
        self.play(FadeIn(final_output))
        self.wait(1)
