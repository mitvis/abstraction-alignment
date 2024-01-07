export type Node = {
    name: string,
    id: number,
    parent: number | null,
    value: number | null
  }
  
export type Tree = Node[]

export type Score = {
    [key: string]: number;
}

export type VisualizationAttribute = {
    id: number,
    name: string,
    isCollapsed: boolean, 
    isVisible: boolean, 
    yIndex: number, 
    x0: number
}

export type VisualizationAttributes = {
    [key: number]: VisualizationAttribute
}

export type Embedding = number[]

export type EmbeddingPoint = {
    x: number,
    y: number,
    id: number,
}