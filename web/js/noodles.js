const { app } = window.comfyAPI.app;

app.registerExtension({
  name: "Noodles.ULIDPreviewNood",

  async beforeRegisterNodeDef(nodeType, nodeData, comfApp) {
    if (nodeData.name === "ULIDPreviewNood") {
      let onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated && onNodeCreated.apply(this, arguments);

        const pw = app.widgets["MARKDOWN"](this, "preview", ["MARKDOWN", { multiline: true }], app).widget;

        pw.options.read_only = true;
        pw.element.readOnly = true;
        pw.element.rows = 1;

        pw.y = 21;
        pw.serialize = false;
      };

      let onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted && onExecuted.apply(this, arguments);
        const pw = this.widgets?.find((w) => w.name === "preview");
        if (pw) {
          pw.value = message.text[0];
        }
      };
    }
  },
});
