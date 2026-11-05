// WebGL2 overlay. The whole timeline is uploaded once as two texture arrays (one layer per
// tubelet), so scrubbing only changes a uniform and never re-uploads data.

const VERT = `#version 300 es
in vec2 pos;
out vec2 uv;
void main() {
  uv = vec2(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
  gl_Position = vec4(pos, 0.0, 1.0);
}`

const FRAG = `#version 300 es
precision highp float;
precision highp sampler2DArray;
uniform sampler2DArray pcaTex;
uniform sampler2DArray surpriseTex;
uniform float layer;
uniform int mode;      // 0 = pca, 1 = surprise
uniform float opacity;
in vec2 uv;
out vec4 color;

// Polynomial fit of matplotlib's inferno colormap.
vec3 inferno(float t) {
  const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
  const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
  const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
  const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
  const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
  const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
  const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

void main() {
  if (mode == 0) {
    color = vec4(texture(pcaTex, vec3(uv, layer)).rgb, opacity);
  } else {
    float s = texture(surpriseTex, vec3(uv, layer)).r;
    // Low surprise fades out so the video shows through where nothing unexpected happens.
    color = vec4(inferno(clamp(s, 0.0, 1.0)), opacity * smoothstep(0.05, 0.6, s));
  }
}`

export interface OverlayState {
  layer: number
  mode: 'pca' | 'surprise'
  opacity: number
  smooth: boolean
}

export class Overlay {
  private gl: WebGL2RenderingContext
  private program: WebGLProgram
  private pcaTex: WebGLTexture
  private surpriseTex: WebGLTexture
  private loc: Record<string, WebGLUniformLocation | null> = {}
  private canvas: HTMLCanvasElement

  constructor(
    canvas: HTMLCanvasElement,
    count: number,
    gridW: number,
    gridH: number,
    pca: Uint8Array,
    surprise: Uint8Array,
  ) {
    this.canvas = canvas
    const gl = canvas.getContext('webgl2', { premultipliedAlpha: false, alpha: true })
    if (!gl) throw new Error('WebGL2 is not available in this browser')
    this.gl = gl
    this.program = link(gl, VERT, FRAG)
    gl.useProgram(this.program)
    for (const name of ['pcaTex', 'surpriseTex', 'layer', 'mode', 'opacity']) {
      this.loc[name] = gl.getUniformLocation(this.program, name)
    }

    const buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW)
    const posLoc = gl.getAttribLocation(this.program, 'pos')
    gl.enableVertexAttribArray(posLoc)
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0)

    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
    this.pcaTex = upload(gl, 0, gl.RGB8, gl.RGB, gridW, gridH, count, pca)
    this.surpriseTex = upload(gl, 1, gl.R8, gl.RED, gridW, gridH, count, surprise)
    gl.uniform1i(this.loc.pcaTex, 0)
    gl.uniform1i(this.loc.surpriseTex, 1)
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
  }

  draw(s: OverlayState) {
    const { gl, canvas } = this
    const dpr = window.devicePixelRatio || 1
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr))
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr))
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w
      canvas.height = h
    }
    gl.viewport(0, 0, w, h)
    gl.clearColor(0, 0, 0, 0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    if (s.layer < 0) return

    const filter = s.smooth ? gl.LINEAR : gl.NEAREST
    for (const [unit, tex] of [[0, this.pcaTex], [1, this.surpriseTex]] as const) {
      gl.activeTexture(gl.TEXTURE0 + unit)
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex)
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, filter)
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, filter)
    }
    gl.uniform1f(this.loc.layer, s.layer)
    gl.uniform1i(this.loc.mode, s.mode === 'pca' ? 0 : 1)
    gl.uniform1f(this.loc.opacity, s.opacity)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
  }
}

function upload(
  gl: WebGL2RenderingContext,
  unit: number,
  internal: number,
  format: number,
  w: number,
  h: number,
  layers: number,
  data: Uint8Array,
): WebGLTexture {
  const tex = gl.createTexture()
  if (!tex) throw new Error('could not create texture')
  gl.activeTexture(gl.TEXTURE0 + unit)
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex)
  gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, internal, w, h, layers, 0, format, gl.UNSIGNED_BYTE, data)
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  return tex
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const compile = (type: number, src: string) => {
    const sh = gl.createShader(type)!
    gl.shaderSource(sh, src)
    gl.compileShader(sh)
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) ?? 'shader error')
    return sh
  }
  const p = gl.createProgram()!
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs))
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs))
  gl.linkProgram(p)
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? 'link error')
  return p
}
