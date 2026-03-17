# llmuxer-egui

egui widget for configuring llmuxer providers at runtime. Renders a settings button that opens a modal panel with per-provider fields. Changes are committed to `llmuxer-keystore` on Save.

## Usage

Add `LlmConfigWidget` to your `eframe::App` struct and call `show` every frame:

```rust
use llmuxer_egui::LlmConfigWidget;
use llmuxer::LlmConfig;

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

The widget loads saved credentials from `llmuxer-keystore` on construction and reloads them on Cancel.
