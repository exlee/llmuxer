# llmux-egui

egui widget for configuring llmux providers at runtime. Renders a settings button that opens a modal panel with per-provider fields. Changes are committed to `llmux-keystore` on Save.

## Usage

Add `LlmConfigWidget` to your `eframe::App` struct and call `show` every frame:

```rust
use llmux_egui::LlmConfigWidget;
use llmux::LlmConfig;

struct MyApp {
    llm_widget: LlmConfigWidget,
    llm_config: Option<LlmConfig>,
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(config) = self.llm_widget.show(ctx, ui) {
                self.llm_config = Some(config);
            }
        });
    }
}

impl MyApp {
    fn new() -> Self {
        Self {
            llm_widget: LlmConfigWidget::new(None),
            llm_config: None,
        }
    }
}
```

The widget loads saved credentials from `llmux-keystore` on construction and reloads them on Cancel.
